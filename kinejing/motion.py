"""CPU nonparametric action-conditioned video predictor, exploratory KW-TWB-0001.

Public validation episodes are the example bank. Retrieval uses only first RGB,
instruction and supplied action trajectories. No held-out/test future is used
to select its donor. All three cameras share one donor and one alignment.
"""
import argparse
import json
import re
from pathlib import Path
import concurrent.futures
import subprocess
import cv2
import h5py
import numpy as np

VIEWS = ('head', 'left', 'right')
K = 17
cv2.setNumThreads(1)

def regularize_flow(flow,kind):
    if kind=='flow_smooth': return cv2.GaussianBlur(flow,(0,0),12)
    if kind=='flow_affine':
        grid=np.stack(np.meshgrid(np.arange(320),np.arange(240)),axis=-1).astype(np.float32)
        pts=grid[8:-8:12,8:-8:12].reshape(-1,2)
        dst=(grid+flow)[8:-8:12,8:-8:12].reshape(-1,2)
        matrix,_=cv2.estimateAffinePartial2D(pts,dst,method=cv2.RANSAC,ransacReprojThreshold=5)
        if matrix is None: return np.zeros_like(flow)
        if not .5<float(np.linalg.det(matrix[:,:2]))<2: return np.zeros_like(flow)
        mapped=grid@matrix[:,:2].T+matrix[:,2]
        return (mapped-grid).astype(np.float32)
    return flow

def read_input(root, n):
    with h5py.File(root/'data'/f'episode{n}.hdf5', 'r') as f:
        q = f['joint_action/vector'][:].astype(np.float32)
    ix = np.linspace(0, len(q)-1, K).round().astype(int)
    images = np.stack([cv2.imread(str(root/'first_frame'/f'episode{n}_{v}.jpg')) for v in VIEWS])
    assert images.shape == (3, 240, 320, 3), images.shape
    prompt = json.loads((root/'instructions'/f'episode{n}.json').read_text(encoding='utf-8'))['instruction']
    return {'id': n, 'q': q, 'qs': q[ix], 'images': images, 'prompt': prompt,
            'small': np.stack([cv2.resize(x, (32,24)) for x in images]).astype(np.float32)/255,
            'indices': ix}

def tokens(prompt):
    stop = {'the','a','an','and','to','with','of','it','on','in','then','from','using'}
    return set(re.findall('[a-z]+', prompt.lower()))-stop

def choose(target, bank):
    costs = []
    for donor in bank:
        qa = target['qs'] - target['qs'][0]
        qb = donor['qs'] - donor['qs'][0]
        action = float(np.mean((qa-qb)**2))
        visual = float(np.mean(np.abs(target['small']-donor['small'])))
        a,b = tokens(target['prompt']),tokens(donor['prompt'])
        language = 1 - len(a&b)/max(1,len(a|b))
        costs.append(action + 0.5*visual + 0.18*language)
    idx = int(np.argmin(costs))
    return bank[idx], costs[idx]

def align(target, donor):
    # One monotone path shared by the three views; inputs contain the full actions.
    a=target['qs']-target['qs'][0]
    b=donor['q']-donor['q'][0]
    cost=np.mean((a[:,None,:]-b[None,:,:])**2,axis=-1)
    cost += 0.025*(np.linspace(0,1,K)[:,None]-np.linspace(0,1,len(b))[None,:])**2
    path=np.zeros(K,dtype=int)
    dp=cost[0].copy(); dp[1:]=1e9
    back=np.zeros((K,len(b)),dtype=int)
    for i in range(1,K):
        best=0
        for j in range(len(b)):
            if dp[j]<dp[best]: best=j
            back[i,j]=best
        dp=cost[i]+dp[back[i]]
    path[-1]=int(np.argmin(dp))
    for i in range(K-1,0,-1): path[i-1]=back[i,path[i]]
    return path

def frames(root, n, view, indices):
    folder=root/'gt_dataset'/f'episode{n}'/view/'frames'
    result=[]
    for i in indices:
        im=cv2.imread(str(folder/f'frame_{int(i):05d}.jpg'))
        if im is None: raise FileNotFoundError(folder/f'frame_{int(i):05d}.jpg')
        result.append(im)
    return np.stack(result)

def predict(initial, donor_frames, kind):
    if kind=='static': return np.repeat(initial[None],len(donor_frames),axis=0)
    if kind.startswith('residual'):
        alpha=float(kind.split('_')[1])
        delta=donor_frames.astype(np.float32)-donor_frames[:1].astype(np.float32)
        return np.clip(initial[None].astype(np.float32)+alpha*delta,0,255).astype(np.uint8)
    if kind == 'foreground':
        first=cv2.cvtColor(donor_frames[0],cv2.COLOR_BGR2GRAY).astype(np.float32)
        out=[initial]
        for im in donor_frames[1:]:
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
            mask=np.clip((first-gray-20)/35,0,1)*np.clip((105-gray)/35,0,1)
            mask=cv2.GaussianBlur(mask,(5,5),0.8)[:,:,None]
            out.append(np.clip(initial.astype(np.float32)*(1-mask)+im*mask,0,255).astype(np.uint8))
        return np.stack(out)
    if kind.startswith('flow'):
        first=cv2.cvtColor(donor_frames[0],cv2.COLOR_BGR2GRAY)
        dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        grid=np.stack(np.meshgrid(np.arange(320),np.arange(240)),axis=-1).astype(np.float32)
        out=[initial]
        for im in donor_frames[1:]:
            flow=dis.calc(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),first,None)
            flow=regularize_flow(flow,kind)
            mapped=grid+flow
            out.append(cv2.remap(initial,mapped[:,:,0],mapped[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101))
        return np.stack(out)
    raise ValueError(kind)

def metrics(pred, gt):
    psnr=[];ssim=[]
    for p,g in zip(pred[1:],gt[1:]):
        a=p.astype(np.float32);b=g.astype(np.float32)
        mse=np.mean((a-b)**2)
        psnr.append(float(10*np.log10(255**2/max(mse,1e-9))))
        a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY);b=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
        ma=cv2.GaussianBlur(a,(11,11),1.5);mb=cv2.GaussianBlur(b,(11,11),1.5)
        va=cv2.GaussianBlur(a*a,(11,11),1.5)-ma*ma
        vb=cv2.GaussianBlur(b*b,(11,11),1.5)-mb*mb
        cov=cv2.GaussianBlur(a*b,(11,11),1.5)-ma*mb
        s=((2*ma*mb+6.5025)*(2*cov+58.5225))/((ma*ma+mb*mb+6.5025)*(va+vb+58.5225))
        ssim.append(float(s[5:-5,5:-5].mean()))
    return {'psnr':float(np.mean(psnr)),'ssim':float(np.mean(ssim)),
            'motion':float(np.mean(np.abs(np.diff(pred.astype(np.float32),axis=0))))}

def evaluate(root, out):
    all_inputs=[read_input(root/'test_dataset',n) for n in range(1,101)]
    bank=[x for x in all_inputs if x['id']%5]
    held=[x for x in all_inputs if not x['id']%5]
    kinds=['static','residual_0.25','residual_0.5','residual_0.75','flow','foreground','flow_smooth','flow_affine']
    records=[]
    for target in held:
        donor,cost=choose(target,bank); path=align(target,donor)
        row={'target':target['id'],'donor':donor['id'],'retrieval_cost':cost,'views':{}}
        for vi,view in enumerate(VIEWS):
            gt=frames(root,target['id'],view,target['indices'])
            df=frames(root,donor['id'],view,path)
            row['views'][view]={kind:metrics(predict(target['images'][vi],df,kind),gt) for kind in kinds}
        records.append(row)
        print('Evaluated',target['id'],'donor',donor['id'],flush=True)
    summary={};selected={}
    for view in VIEWS:
        summary[view]={}
        for kind in kinds:
            pairs={m:float(np.median([r['views'][view][kind][m]-r['views'][view]['static'][m] for r in records])) for m in ('psnr','ssim')}
            summary[view][kind]=pairs
        eligible=[k for k in kinds if summary[view][k]['psnr']>0 and summary[view][k]['ssim']>0]
        selected[view]=max(eligible,key=lambda k:summary[view][k]['ssim']) if eligible else 'static'
    report={'evidence':'exploratory held-out internal comparison, not TWB-Score','heldout':[x['id'] for x in held],
            'candidates':kinds,'summary_median_paired_delta':summary,'selected':selected,'records':records}
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'summary':summary,'selected':selected},indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--validation',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();evaluate(a.validation,a.report)

if __name__=='__main__': main()
