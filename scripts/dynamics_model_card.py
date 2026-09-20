"""Generate the neural experiment model card from its machine-readable record."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def document():
    r=json.loads((ROOT/'evidence/dynamics_v01.json').read_text(encoding='utf-8'))
    split=r['split_episode_ids']
    assert not (set(split['train'])&set(split['validation']) or set(split['train'])&set(split['evaluation']) or set(split['validation'])&set(split['evaluation']))
    rows=r['metrics']['evaluation']
    lookup={x['method']:x['mse'] for x in rows}
    gain=100*(1-lookup['learned']/lookup['persistence'])
    lines=['# KineJing Dynamics v0.1','',
        'A trained KineWorld action-conditioned three-camera **latent predictor** on top of frozen official DINOv2 ViT-S/14. This does not generate RGB videos and is not a new foundation model.', '',
        f"Frozen encoder parameters: {r['encoder']['frozen_encoder_parameters']:,}. Trainable KineWorld parameters: {r['trainable_parameters']:,}.",
        f"Training / checkpoint selection / local evaluation episodes: {len(split['train'])} / {len(split['validation'])} / {len(split['evaluation'])}. Selected epoch: {r['selected_epoch']}. Seed: {r['seed']}.", '',
        '## Local measurements','', '| Method | Normalized latent MSE (lower is better) |','|---|---:|']
    for row in rows: lines.append(f"| {row['method']} | {row['mse']:.6f} |")
    lines += ['',f'Relative error reduction against first-frame persistence: **{gain:.2f}%**.',
        'This number measures frozen-encoder features only. It is not a TWB score, RGB quality improvement, comparison with CausalWM, or proof of generalization to new tasks.', '',
        'The 100 public development examples have already been explored by earlier CPU experiments; this is not blind testing. Split is by episode. Only training episodes provide normalization statistics. Checkpoint selection uses the separate validation split; the final split is evaluated once. One action shuffle is a sensitivity diagnostic, not causal identification.', '',
        '## Reproduce','',
        'Use Python 3.10+, PyTorch with a matching CUDA runtime, torchvision, OpenCV, NumPy and h5py. The recorded environment is in the JSON report. Obtain the original data and DINOv2 weights separately under their terms. Upstream encoder weights and benchmark data are not redistributed.', '',
        '```bash',
        'git clone https://github.com/facebookresearch/dinov2.git external/dinov2',
        f"git -C external/dinov2 checkout {r['encoder']['revision']}",
        'python scripts/train_dynamics.py --dataset data/validation --dino-source external/dinov2 --encoder-weights checkpoints/dinov2_vits14_pretrain.pth --output runs/dynamics-v01',
        'python scripts/predict_dynamics.py --inputs data/test_dataset --episode 1 --checkpoint runs/dynamics-v01/kinejing-dynamics.pt --dino-source external/dinov2 --encoder-weights checkpoints/dinov2_vits14_pretrain.pth --output runs/episode1-latents.npz',
        '```','',
        'The inference script reads only three initial images and the supplied joint trajectory. Output shape is `[batch, sampled_time, head/left/right, feature]`. It checks the encoder weight hash and loads model tensors strictly. It does not consume future frames.', '',
        'Action convention: unchanged `joint_action/vector` order, dataset-native units. This is not a generic robotics action protocol. Nine uniformly sampled time points omit intervening motion. CLS and mean patch tokens discard detailed spatial structure.', '',
        '## Artifacts and boundaries','',
        f"Checkpoint SHA-256: `{r['checkpoint_sha256']}`.",
        f"Encoder SHA-256: `{r['encoder']['weights_sha256']}`.",
        '[Full results and training history](../evidence/dynamics_v01.json) · [Preregistered protocol](../experiments/KJ-DINO-0001.md).', '',
        'Research checkpoint is retained in the private company knowledge release `dynamics-v0.1.0`; company members can retrieve it without putting weights in Git history. No upstream model checkpoint is included.', '',
        'CausalWM / Wan / V-JEPA generation has not been run or jointly trained with this predictor. The separate [CausalWM derivative](https://github.com/kineworld/KineJing-CausalWM) changes VAE decoding only; it has no KineJing quality score.', '']
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    path=ROOT/'docs/DYNAMICS_MODEL_CARD.md';expected=document()
    if a.write:path.write_text(expected,encoding='utf-8')
    assert path.read_text(encoding='utf-8')==expected
    print('Dynamics model card matches recorded metrics.')
