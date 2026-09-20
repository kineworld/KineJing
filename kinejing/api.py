"""Local authenticated research API; outputs latent states, not video or control."""
import base64
import binascii
from contextlib import asynccontextmanager
import hashlib
import hmac
import io
import os
from pathlib import Path
import threading
import time
from typing import Literal
import uuid

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action_schema: Literal['triworldbench-joint-action-vector-v1']
    images: dict[str, str]
    actions: list[list[FiniteFloat]] = Field(min_length=9, max_length=4096)

    @field_validator('images')
    @classmethod
    def three_views(cls, value):
        if set(value) != {'head','left','right'}:
            raise ValueError('Exactly head, left and right base64 image strings are required')
        if any(not s or len(s)>1400000 for s in value.values()):
            raise ValueError('Each encoded image must be nonempty and at most 1.4M characters')
        return value

    @field_validator('actions')
    @classmethod
    def action_columns(cls, value):
        if any(len(row)!=14 for row in value):
            raise ValueError('Each action row must have exactly 14 dataset-native columns')
        if any(abs(x)>1e30 for row in value for x in row):
            raise ValueError('Action value exceeds supported numeric range')
        return value


def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class DynamicsEngine:
    def __init__(self, checkpoint, encoder_weights, dino_source, device='cuda'):
        import torch
        from .learned_dynamics import TriViewDynamics
        torch.set_num_threads(2)
        self.torch=torch; self.device=torch.device(device)
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if file_sha(encoder_weights)!=state['encoder_provenance']['weights_sha256']:
            raise ValueError('Encoder weight hash differs from the training feature space')
        self.state=state;self.checkpoint_sha256=file_sha(checkpoint)
        self.encoder=torch.hub.load(str(dino_source),'dinov2_vits14',source='local',pretrained=False)
        self.encoder.load_state_dict(torch.load(encoder_weights,map_location='cpu',weights_only=True),strict=True)
        self.encoder.to(self.device).eval().requires_grad_(False)
        self.model=TriViewDynamics(**state['config']).to(self.device).eval().requires_grad_(False)
        self.model.load_state_dict(state['state_dict'],strict=True)

    def predict(self, request):
        import cv2
        import numpy as np
        from PIL import Image
        torch=self.torch; device=self.device;state=self.state
        ims=[]
        for view in ('head','left','right'):
            try:
                data=base64.b64decode(request.images[view],validate=True)
                if len(data)>1024*1024:raise ValueError('Image exceeds one MiB')
                with Image.open(io.BytesIO(data)) as image:
                    if image.format not in ('JPEG','PNG') or getattr(image,'n_frames',1)!=1:
                        raise ValueError('Use a single JPEG or PNG observation')
                    if min(image.size)<1 or max(image.size)>2048:
                        raise ValueError('Image dimensions must be between 1 and 2048')
                    image.verify()
                im=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
                if im is None:raise ValueError('Cannot decode observation')
                ims.append(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(224,224)))
            except (ValueError,OSError,binascii.Error,Image.DecompressionBombError) as exc:
                raise ValueError('Invalid '+view+' observation') from exc
        q=np.asarray(request.actions,dtype=np.float32)
        indices=np.linspace(0,len(q)-1,9).round().astype(int)
        mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
        std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
        x=torch.from_numpy(np.stack(ims)).to(device).permute(0,3,1,2).float()/255
        with torch.inference_mode():
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                features=self.encoder.forward_features((x-mean)/std)
            z=torch.cat((features['x_norm_clstoken'],features['x_norm_patchtokens'].mean(1)),-1).float()[None]
            fm=state['feature_mean'].to(device)[:,0];fs=state['feature_std'].to(device)[:,0]
            qa=(torch.tensor(q[indices][None],device=device)-state['action_mean'].to(device))/state['action_std'].to(device)
            t=torch.tensor(indices/(len(q)-1),device=device,dtype=torch.float32)[None,:,None]
            predictions=self.model((z-fm)/fs,qa,t)*fs[:,None]+fm[:,None]
            if not bool(torch.isfinite(predictions).all()):raise RuntimeError('Nonfinite model output')
            values=predictions[0].cpu().tolist()
        return {'model':'kinejing-dynamics-v0.1','output_kind':'predicted_latents_not_rgb',
                'shape':[9,3,768],'views':['head','left','right'],'frame_indices':indices.tolist(),
                'predicted_features':values,'checkpoint_sha256':self.checkpoint_sha256}


class AdmissionMiddleware:
    """Authenticate before reading bodies and cap buffered request bytes."""
    def __init__(self, app, key, max_bytes=4*1024*1024):
        self.app=app;self.key=('Bearer '+key).encode();self.max_bytes=max_bytes

    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        async def reject(code,body):
            await send({'type':'http.response.start','status':code,'headers':[(b'content-type',b'application/json')]})
            await send({'type':'http.response.body','body':body})
        if scope['path'].startswith('/v1/'):
            headers=dict(scope['headers'])
            if not hmac.compare_digest(headers.get(b'authorization',b''),self.key):
                return await reject(401,b'{"detail":"Invalid API key"}')
        if scope['method']=='POST':
            chunks=[];size=0
            while True:
                message=await receive()
                if message['type']=='http.disconnect':return
                chunk=message.get('body',b'');size+=len(chunk)
                if size>self.max_bytes:return await reject(413,b'{"detail":"Request too large"}')
                chunks.append(chunk)
                if not message.get('more_body',False):break
            body=b''.join(chunks);sent=False
            async def bounded_receive():
                nonlocal sent
                if not sent:
                    sent=True
                    return {'type':'http.request','body':body,'more_body':False}
                return await receive()
            return await self.app(scope,bounded_receive,send)
        return await self.app(scope,receive,send)


def create_app(engine_factory=None, api_key=None):
    key=api_key or os.environ.get('KINEJING_API_KEY','')
    if len(key)<24:raise ValueError('Set KINEJING_API_KEY to at least 24 characters')
    lock=threading.Lock()
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine=engine_factory() if engine_factory else DynamicsEngine(
            Path(os.environ['KINEJING_CHECKPOINT']),Path(os.environ['KINEJING_ENCODER_WEIGHTS']),
            Path(os.environ['KINEJING_DINO_SOURCE']),os.environ.get('KINEJING_DEVICE','cuda'))
        yield
        del app.state.engine
    app=FastAPI(title='KineJing Research API',version='0.1.0',lifespan=lifespan,
                description='Local research preview. Predicts latent states; no video generation, billing or robot control.')
    app.add_middleware(AdmissionMiddleware,key=key)
    bearer=HTTPBearer()
    @app.get('/health')
    def health():return {'ready':hasattr(app.state,'engine'),'mode':'local_research_preview'}
    @app.get('/v1/models',dependencies=[Depends(bearer)])
    def models():return {'models':[{'id':'kinejing-dynamics-v0.1','output':'latent_features','commercial_ready':False}]}
    @app.post('/v1/predictions',dependencies=[Depends(bearer)])
    def predict(request:PredictionRequest):
        if not lock.acquire(blocking=False):raise HTTPException(429,'Model is busy; retry later')
        try:
            started=time.perf_counter()
            result=app.state.engine.predict(request)
            result.update(request_id=str(uuid.uuid4()),inference_seconds=time.perf_counter()-started)
            return result
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc
        except Exception as exc:raise HTTPException(503,'Inference failed; no result produced') from exc
        finally:lock.release()
    return app


if __name__=='__main__':
    import uvicorn
    uvicorn.run(create_app(),host='127.0.0.1',port=int(os.environ.get('KINEJING_PORT','8765')),
                workers=1,limit_concurrency=8,timeout_keep_alive=5,access_log=False)
