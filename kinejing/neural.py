"""Optional worker: run in the separately installed upstream Torch environment."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    if sys.version_info<(3,10):raise RuntimeError('Optional neural workers require Python 3.10+ in their upstream environment.')
    p=argparse.ArgumentParser();p.add_argument('backend',choices=['vjepa21','kine-jepa'])
    for k in ['repo','checkpoint','input','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',default='cpu');a=p.parse_args()
    import torch
    if a.output.exists():raise FileExistsError(a.output)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if a.backend=='vjepa21':
        import cv2
        # Explicit local checkpoint avoids the upstream hub's current localhost download URL.
        encoder,predictor=torch.hub.load(str(a.repo),'vjepa2_1_vit_base_384',source='local',pretrained=False)
        del predictor
        raw=state['ema_encoder'];clean={k.removeprefix('module.').removeprefix('backbone.'):v for k,v in raw.items()}
        encoder.load_state_dict(clean,strict=True);encoder=encoder.to(a.device).eval()
        transform=torch.hub.load(str(a.repo),'vjepa2_preprocessor',source='local',crop_size=384)
        cap=cv2.VideoCapture(str(a.input));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count<1:raise ValueError('Cannot decode input video')
        frames=[]
        for i in np.linspace(0,count-1,64).round().astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,im=cap.read()
            if not ok:raise ValueError('Failed to decode frame '+str(i))
            frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        cap.release();video=transform(np.stack(frames)).unsqueeze(0).to(a.device)
        with torch.inference_mode():tokens=encoder(video)
        if not isinstance(tokens,torch.Tensor) or tokens.ndim!=3:raise ValueError('Unexpected V-JEPA token layout')
        tokens=tokens.float().cpu().numpy();meta={'encoder':'vjepa2_1_vit_base_384','checkpoint_sha256':digest(a.checkpoint)}
        extra={}
    else:
        spec=importlib.util.spec_from_file_location('kinejing_external_rollout',a.repo/'kineworld_jepa/rollout.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        config=state['config'];model=module.ActionRollout(**config)
        model.load_state_dict(state['state_dict'],strict=True);model=model.to(a.device).eval()
        with np.load(a.input,allow_pickle=False) as b:
            latent=b['tokens'];actions=b['actions'];meta=json.loads(str(b['metadata'].item()))
        if latent.ndim!=3 or actions.ndim!=3 or actions.shape[1]<1 or latent.shape[0]!=actions.shape[0]:raise ValueError('Expected tokens B,N,D and actions B,H,A')
        if latent.shape[-1]!=config['dim'] or actions.shape[-1]!=config.get('action_dim',8):raise ValueError('Checkpoint dimensions do not match inputs')
        if state.get('encoder_provenance')!=meta:raise ValueError('Rollout checkpoint was not trained in the supplied feature space')
        if not np.isfinite(latent).all() or not np.isfinite(actions).all():raise ValueError('Non-finite input')
        with torch.inference_mode():
            sequence=model(torch.as_tensor(latent,device=a.device,dtype=torch.float32),torch.as_tensor(actions,device=a.device,dtype=torch.float32))
        rollout=np.stack([x.float().cpu().numpy() for x in sequence]);tokens=rollout[-1]
        extra={'rollout':rollout,'rollout_checkpoint_sha256':np.array(digest(a.checkpoint))}
    if not np.isfinite(tokens).all():raise ValueError('Model produced non-finite features')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('wb') as f:np.savez_compressed(f,tokens=tokens,embedding=tokens.mean(axis=(0,1)),metadata=np.array(json.dumps(meta)),**extra)

if __name__=='__main__':main()
