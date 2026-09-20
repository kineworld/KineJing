"""Second exploratory development round; never uses target future for prediction."""
import argparse,json,concurrent.futures
from pathlib import Path
import cv2,numpy as np
from .motion import read_input,choose,align,frames,metrics,regularize_flow,VIEWS,K

GRID=np.stack(np.meshgrid(np.arange(320),np.arange(240)),axis=-1).astype(np.float32)

def flow_knots(initial,df,mix):
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    first=cv2.cvtColor(df[0],cv2.COLOR_BGR2GRAY)
    result=[np.zeros((240,320,2),np.float32)]
    for im in df[1:]:
        raw=dis.calc(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),first,None)
        rigid=regularize_flow(raw,'flow_affine')
        smooth=regularize_flow(raw,'flow_smooth')
        # Bounded local correction retains global rigid motion and limits distortion.
        delta=smooth-rigid
        norm=np.linalg.norm(delta,axis=-1,keepdims=True)
        delta*=np.minimum(1,8/np.maximum(norm,1e-6))
        result.append(rigid+mix*delta)
    return np.stack(result)

def render(initial,knots):
    return np.stack([cv2.remap(initial,(GRID+k)[:,:,0],(GRID+k)[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101) for k in knots])

def gain(target,donor,path,vi):
    sl=slice(0,6) if vi==1 else slice(7,13)
    a=target['qs'][:,sl]-target['qs'][0,sl]
    b=donor['q'][path,sl]-donor['q'][0,sl]
    # Displacement ratio attenuates inactive arms; amplification capped at 1.
    ratio=np.clip(np.linalg.norm(a,axis=1)/np.maximum(np.linalg.norm(b,axis=1),.03),0,1)
    ratio[0]=0
    return ratio[:,None,None,None]

def episode(root,n,bank):
    target=read_input(root/'test_dataset',n);donor,cost=choose(target,bank);path=align(target,donor)
    row={'episode':n,'donor':donor['id'],'views':{}}
    for vi,view in enumerate(VIEWS):
        df=frames(root,donor['id'],view,path);gt=frames(root,n,view,target['indices'])
        base=flow_knots(target['images'][vi],df,0)
        corrected=flow_knots(target['images'][vi],df,1)
        if vi==0:
            variants={'baseline':np.zeros_like(base),'head_bounded':corrected*.25}
        else:
            g=gain(target,donor,path,vi)
            variants={'baseline':base,'gated':base*g,'bounded':corrected,'bounded_2px':base+.25*(corrected-base),'bounded_gated':corrected*g}
        row['views'][view]={kind:metrics(render(target['images'][vi],f),gt) for kind,f in variants.items()}
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--confirm',action='store_true');a=p.parse_args()
    held=list(range(1,101,5)) if a.confirm else list(range(5,101,5))
    bank=[read_input(a.root/'test_dataset',n) for n in range(1,101) if n%5 and n not in held];rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(episode,a.root,n,bank) for n in held]
        for job in concurrent.futures.as_completed(jobs):
            rows.append(job.result());print('Evaluated',len(rows),'/20',flush=True)
    summary={}
    for view in VIEWS:
        summary[view]={}
        for kind in rows[0]['views'][view]:
            summary[view][kind]={m:{'median_delta':float(np.median([r['views'][view][kind][m]-r['views'][view]['baseline'][m] for r in rows])), 'mean_delta':float(np.mean([r['views'][view][kind][m]-r['views'][view]['baseline'][m] for r in rows]))} for m in ['psnr','ssim']}
    a.report.write_text(json.dumps({'evidence':'Additional 20 targets excluded from example bank; only preselected bounded_2px is the confirmation candidate' if a.confirm else 'Exploratory reused 20-episode development holdout, not official TWB score','bank_ids':[x['id'] for x in bank],'heldout':held,'summary':summary,'records':sorted(rows,key=lambda r:r['episode'])},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
