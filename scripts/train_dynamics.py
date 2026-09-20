"""Reproduce KJ-DINO-0001 with explicit local source, weights and public data."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinejing.learned_dynamics import TriViewDynamics

VIEWS = ('head', 'left', 'right')
SEED = 20260920
REV = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def extract(args, device):
    cache = args.output / 'features.npz'
    if cache.exists():
        raise FileExistsError('Use a fresh output directory; cached features must not be silently reused')
    encoder = torch.hub.load(str(args.dino_source), 'dinov2_vits14', source='local', pretrained=False)
    encoder.load_state_dict(torch.load(args.encoder_weights, map_location='cpu', weights_only=True), strict=True)
    encoder = encoder.to(device).eval().requires_grad_(False)
    arrays, actions, times = [], [], []
    data_hash = hashlib.sha256()
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    for episode in range(1, 101):
        action_file = args.dataset / 'test_dataset' / 'data' / f'episode{episode}.hdf5'
        data_hash.update(sha(action_file).encode())
        with h5py.File(action_file) as f:
            q = f['joint_action/vector'][:].astype(np.float32)
        if q.ndim != 2 or q.shape[1] != 14 or len(q) < 9 or not np.isfinite(q).all():
            raise ValueError(f'Invalid actions in episode {episode}')
        indices = np.linspace(0, len(q)-1, 9).round().astype(int)
        images = []
        for index in indices:
            for view in VIEWS:
                p = args.dataset / 'gt_dataset' / f'episode{episode}' / view / 'frames' / f'frame_{index:05d}.jpg'
                data_hash.update(sha(p).encode())
                im = cv2.imread(str(p))
                if im is None:
                    raise ValueError(f'Missing image: {p.name}')
                images.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (224, 224)))
        encoded = []
        with torch.inference_mode():
            for start in range(0, len(images), 9):
                x = torch.from_numpy(np.stack(images[start:start+9])).to(device).permute(0, 3, 1, 2).float()/255
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=torch.float16):
                    y = encoder.forward_features((x-mean)/std)
                encoded.append(torch.cat((y['x_norm_clstoken'], y['x_norm_patchtokens'].mean(1)), -1).float().cpu().numpy())
        arrays.append(np.concatenate(encoded).reshape(9, 3, 768))
        actions.append(q[indices])
        times.append((indices/(len(q)-1)).astype(np.float32))
        if episode % 10 == 0:
            print(f'Encoded {episode}/100 episodes', flush=True)
    features = np.stack(arrays)
    actions = np.stack(actions)
    times = np.stack(times)[..., None]
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite encoder features')
    np.savez_compressed(cache, features=features, actions=actions, times=times)
    provenance = {'repository': 'https://github.com/facebookresearch/dinov2',
                  'revision': REV, 'architecture': 'dinov2_vits14',
                  'weights_sha256': sha(args.encoder_weights), 'sampled_data_sha256': data_hash.hexdigest(),
                  'preprocessing': 'OpenCV linear resize RGB 224x224; ImageNet mean/std',
                  'features': 'CLS concatenated with mean patch token; 768 dimensions',
                  'frozen_encoder_parameters': sum(p.numel() for p in encoder.parameters())}
    del encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return features, actions, times, provenance


def train(args):
    start = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(2); cv2.setNumThreads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    z, q, t, provenance = extract(args, device)
    ids = np.arange(1, 101)
    splits = {'train': np.where((ids % 10 != 0) & (ids % 10 != 9))[0],
              'validation': np.where(ids % 10 == 9)[0], 'evaluation': np.where(ids % 10 == 0)[0]}
    tr = splits['train']
    zmean = z[tr].mean(axis=(0, 1, 2), keepdims=True)
    zstd = z[tr].std(axis=(0, 1, 2), keepdims=True).clip(.1)
    qmean = q[tr].mean(axis=(0, 1), keepdims=True)
    qstd = q[tr].std(axis=(0, 1), keepdims=True).clip(.01)
    Z = torch.tensor((z-zmean)/zstd, device=device)
    Q = torch.tensor((q-qmean)/qstd, device=device)
    T = torch.tensor(t, device=device)
    model = TriViewDynamics().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.05)
    best, best_epoch, history = float('inf'), 0, []
    checkpoint = args.output / 'kinejing-dynamics.pt'
    for epoch in range(1, 121):
        model.train(); losses = []
        for batch in np.array_split(np.random.permutation(tr), 10):
            pred = model(Z[batch, 0], Q[batch], T[batch])
            loss = F.mse_loss(pred[:, 1:], Z[batch, 1:])
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vi = splits['validation']
            val = F.mse_loss(model(Z[vi, 0], Q[vi], T[vi])[:, 1:], Z[vi, 1:]).item()
        history.append({'epoch': epoch, 'train_mse': float(np.mean(losses)), 'validation_mse': val})
        if val < best:
            best, best_epoch = val, epoch
            torch.save({'state_dict': {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        'config': model.config, 'encoder_provenance': provenance, 'seed': SEED,
                        'feature_mean': torch.from_numpy(zmean), 'feature_std': torch.from_numpy(zstd),
                        'action_mean': torch.from_numpy(qmean), 'action_std': torch.from_numpy(qstd),
                        'action_schema': {'source': 'TriWorldBench joint_action/vector', 'dimensions': 14,
                                          'column_order': 'unchanged dataset order', 'units': 'dataset-native; not inferred'},
                        'epoch': epoch}, checkpoint)
        if epoch % 20 == 0:
            print(f'Epoch {epoch}/120 validation MSE={val:.6f}, best={best:.6f}', flush=True)
    loaded = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(loaded['state_dict'], strict=True)
    model.eval()
    metrics = {}
    with torch.no_grad():
        for name in ('validation', 'evaluation'):
            ix = splits[name]; target = Z[ix, 1:]
            pred = model(Z[ix, 0], Q[ix], T[ix])[:, 1:]
            static = Z[ix, :1].expand_as(target)
            shuffled = model(Z[ix, 0], Q[np.roll(ix, 1)], T[ix])[:, 1:]
            rows = []
            for method, value in [('learned', pred), ('persistence', static), ('shuffled_actions', shuffled)]:
                errors = ((value-target)**2).mean((1, 3)).cpu().numpy()
                rows.append({'method': method, 'mse': float(errors.mean()),
                             'per_view_mse': dict(zip(VIEWS, map(float, errors.mean(0)))),
                             'per_episode_mse': {str(ids[i]): float(e.mean()) for i,e in zip(ix, errors)}})
            metrics[name] = rows
    report = {'experiment': 'KJ-DINO-0001', 'seed': SEED, 'encoder': provenance,
              'trainable_parameters': sum(p.numel() for p in model.parameters()),
              'split_episode_ids': {k: ids[v].tolist() for k,v in splits.items()},
              'selected_epoch': best_epoch, 'metrics': metrics, 'history': history,
              'checkpoint_sha256': sha(checkpoint), 'duration_seconds': time.time()-start,
              'environment': {'torch': str(torch.__version__), 'numpy': str(np.__version__),
                              'device': torch.cuda.get_device_name() if device.type=='cuda' else 'cpu'},
              'limitations': ['public development set, not a blind benchmark', 'latent prediction only; no RGB decoder',
                              'no official TWB score', 'no CausalWM/Wan/V-JEPA weights used',
                              'sparse action sampling; not robot control', 'single seed']}
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'checkpoint': str(checkpoint), 'selected_epoch': best_epoch, 'metrics': metrics}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--dino-source', type=Path, required=True)
    p.add_argument('--encoder-weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    train(p.parse_args())
