# KineJing Dynamics v0.1

A trained KineWorld action-conditioned three-camera **latent predictor** on top of frozen official DINOv2 ViT-S/14. This does not generate RGB videos and is not a new foundation model.

Frozen encoder parameters: 22,056,576. Trainable KineWorld parameters: 1,341,824.
Training / checkpoint selection / local evaluation episodes: 80 / 10 / 10. Selected epoch: 31. Seed: 20260920.

## Local measurements

| Method | Normalized latent MSE (lower is better) |
|---|---:|
| learned | 0.835667 |
| persistence | 0.991879 |
| shuffled_actions | 0.895649 |

Relative error reduction against first-frame persistence: **15.75%**.
This number measures frozen-encoder features only. It is not a TWB score, RGB quality improvement, comparison with CausalWM, or proof of generalization to new tasks.

The 100 public development examples have already been explored by earlier CPU experiments; this is not blind testing. Split is by episode. Only training episodes provide normalization statistics. Checkpoint selection uses the separate validation split; the final split is evaluated once. One action shuffle is a sensitivity diagnostic, not causal identification.

## Reproduce

Use Python 3.10+, PyTorch with a matching CUDA runtime, torchvision, OpenCV, NumPy and h5py. The recorded environment is in the JSON report. Obtain the original data and DINOv2 weights separately under their terms. Upstream encoder weights and benchmark data are not redistributed.

```bash
git clone https://github.com/facebookresearch/dinov2.git external/dinov2
git -C external/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
python scripts/train_dynamics.py --dataset data/validation --dino-source external/dinov2 --encoder-weights checkpoints/dinov2_vits14_pretrain.pth --output runs/dynamics-v01
python scripts/predict_dynamics.py --inputs data/test_dataset --episode 1 --checkpoint runs/dynamics-v01/kinejing-dynamics.pt --dino-source external/dinov2 --encoder-weights checkpoints/dinov2_vits14_pretrain.pth --output runs/episode1-latents.npz
```

The inference script reads only three initial images and the supplied joint trajectory. Output shape is `[batch, sampled_time, head/left/right, feature]`. It checks the encoder weight hash and loads model tensors strictly. It does not consume future frames.

Action convention: unchanged `joint_action/vector` order, dataset-native units. This is not a generic robotics action protocol. Nine uniformly sampled time points omit intervening motion. CLS and mean patch tokens discard detailed spatial structure.

## Artifacts and boundaries

Checkpoint SHA-256: `f9cf4aeac48ae603c8eedf9f8eb81aa33f939b49406f078f12e2ca9a9c249166`.
Encoder SHA-256: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
[Full results and training history](../evidence/dynamics_v01.json) · [Preregistered protocol](../experiments/KJ-DINO-0001.md).

Research checkpoint is retained in the private company knowledge release `dynamics-v0.1.0`; company members can retrieve it without putting weights in Git history. No upstream model checkpoint is included.

CausalWM / Wan / V-JEPA generation has not been run or jointly trained with this predictor. The separate [CausalWM derivative](https://github.com/kineworld/KineJing-CausalWM) changes VAE decoding only; it has no KineJing quality score.
