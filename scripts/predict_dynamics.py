"""Predict nine future latent states from three initial images and robot actions.

No future images are read. Outputs are DINO features, NOT RGB videos.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinejing.learned_dynamics import TriViewDynamics
from train_dynamics import sha, VIEWS


def predict(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    device = torch.device(a.device)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    provenance = checkpoint['encoder_provenance']
    if sha(a.encoder_weights) != provenance['weights_sha256']:
        raise ValueError('Encoder weights differ from the trained feature space')
    with h5py.File(a.inputs/'data'/f'episode{a.episode}.hdf5') as f:
        q = f['joint_action/vector'][:].astype(np.float32)
    if q.ndim != 2 or q.shape[1] != 14 or len(q) < 9 or not np.isfinite(q).all():
        raise ValueError('Expected finite dataset-native 14-column joint actions, at least nine rows')
    indices = np.linspace(0, len(q)-1, 9).round().astype(int)
    ims = []
    for view in VIEWS:
        im = cv2.imread(str(a.inputs/'first_frame'/f'episode{a.episode}_{view}.jpg'))
        if im is None:
            raise ValueError('Missing first-frame image for '+view)
        ims.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (224,224)))
    encoder = torch.hub.load(str(a.dino_source), 'dinov2_vits14', source='local', pretrained=False)
    encoder.load_state_dict(torch.load(a.encoder_weights, map_location='cpu', weights_only=True), strict=True)
    encoder.to(device).eval().requires_grad_(False)
    model = TriViewDynamics(**checkpoint['config']).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    mean = torch.tensor([.485,.456,.406], device=device)[None,:,None,None]
    std = torch.tensor([.229,.224,.225], device=device)[None,:,None,None]
    x = torch.from_numpy(np.stack(ims)).to(device).permute(0,3,1,2).float()/255
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, enabled=device.type=='cuda', dtype=torch.float16):
            features = encoder.forward_features((x-mean)/std)
        z = torch.cat((features['x_norm_clstoken'],features['x_norm_patchtokens'].mean(1)),dim=-1).float()[None]
        fm = checkpoint['feature_mean'].to(device)[:,0]
        fs = checkpoint['feature_std'].to(device)[:,0]
        qa = (torch.tensor(q[indices][None],device=device)-checkpoint['action_mean'].to(device))/checkpoint['action_std'].to(device)
        t = torch.tensor(indices/(len(q)-1),device=device,dtype=torch.float32)[None,:,None]
        predictions = model((z-fm)/fs,qa,t)*fs[:,None]+fm[:,None]
    if not bool(torch.isfinite(predictions).all()):
        raise ValueError('Nonfinite predictions')
    meta = {'kind':'predicted_latents_not_rgb','checkpoint_sha256':sha(a.checkpoint),
            'encoder_provenance':provenance,'episode':a.episode,'views':list(VIEWS),
            'action_schema':checkpoint['action_schema'],'future_images_read':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,predicted_features=predictions.cpu().numpy(),
                        frame_indices=indices,metadata=np.array(json.dumps(meta)))
    print(json.dumps({'shape':list(predictions.shape),'output':str(a.output),'future_images_read':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--episode',type=int,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dino-source',type=Path,required=True)
    p.add_argument('--encoder-weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    predict(p.parse_args())
