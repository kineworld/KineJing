"""Generate the selected KW-TWB-0001 method on test INPUTS only."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from .motion import VIEWS, K, read_input, choose, align, frames, predict, regularize_flow

def run_episode(n, dataset, validation, bank, selected, output, crf=20, preset='fast'):
    target=read_input(dataset,n)
    donor,cost=choose(target,bank)
    path=align(target,donor)
    assert path[0]==0 and np.all(np.diff(path)>=0)
    dest=output/f'episode{n}';dest.mkdir(parents=True,exist_ok=True)
    grid=np.stack(np.meshgrid(np.arange(320),np.arange(240)),axis=-1).astype(np.float32)
    for vi,view in enumerate(VIEWS):
        kind=selected[view]
        initial=target['images'][vi]
        df=frames(validation,donor['id'],view,path) if kind!='static' else None
        if kind=='flow_bounded_2px':
            from .refinement import flow_knots
            knots=flow_knots(initial,df,.25)
        elif kind.startswith('flow'):
            dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            first=cv2.cvtColor(df[0],cv2.COLOR_BGR2GRAY)
            knots=np.stack([np.zeros((240,320,2),np.float32)]+[
                regularize_flow(dis.calc(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),first,None),kind) for im in df[1:]])
        elif kind=='static':
            knots=np.repeat(initial[None],K,axis=0).astype(np.float32)
        else:
            knots=predict(initial,df,kind).astype(np.float32)
        command=['ffmpeg','-hide_banner','-loglevel','error','-y',
                 '-f','rawvideo','-pix_fmt','bgr24','-s','320x240','-r','30','-i','pipe:0',
                 '-an','-map_metadata','-1','-c:v','libx264','-preset',preset,'-crf',str(crf),
                 '-pix_fmt','yuv420p','-threads','1','-movflags','+faststart',str(dest/f'{view}.mp4')]
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            for t in range(len(target['q'])):
                pos=t/max(1,len(target['q'])-1)*(K-1)
                lo=min(K-2,int(pos));a=pos-lo
                value=knots[lo]*(1-a)+knots[lo+1]*a
                if kind.startswith('flow'):
                    m=grid+value
                    im=cv2.remap(initial,m[:,:,0],m[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
                else: im=np.clip(value,0,255).astype(np.uint8)
                if t==0: im=initial
                process.stdin.write(im.tobytes())
        except BaseException:
            process.kill();process.wait();raise
        finally:
            process.stdin.close()
        error=process.stderr.read().decode(errors='replace')
        if process.wait()!=0: raise RuntimeError(error)
    return {'episode':n,'donor_validation_episode':donor['id'],
            'retrieval_cost':cost,'alignment':path.tolist(),'frames':len(target['q']),
            'source_first_frame_sha256':{
                v:hashlib.sha256((dataset/'first_frame'/f'episode{n}_{v}.jpg').read_bytes()).hexdigest()
                for v in VIEWS}}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int,default=500)
    p.add_argument('--crf',type=int,default=20)
    p.add_argument('--preset',default='fast')
    a=p.parse_args()
    selected=json.loads(a.report.read_text())['selected']
    bank=[read_input(a.validation/'test_dataset',n) for n in range(1,101)]
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(run_episode,n,a.dataset,a.validation,bank,selected,a.output,a.crf,a.preset) for n in range(1,a.limit+1)]
        for done in concurrent.futures.as_completed(jobs):
            records.append(done.result())
            if len(records)%10==0 or len(records)==1: print(f'Generated {len(records)}/{a.limit}',flush=True)
    report={'model':'KineJing','selected':selected,'seed':42,'encoding':{'crf':a.crf,'preset':a.preset},
            'learned_neural_weights':False,'example_bank':'official public 100-episode validation bundle',
            'test_future_frames_used':False,'causalwm_weights_used':False,'jev_used':False,
            'episodes':sorted(records,key=lambda x:x['episode'])}
    a.manifest.write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__': main()
