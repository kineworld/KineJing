# Attribution and component boundaries

This repository contains KineWorld-authored adapters, CPU motion-transfer code,
interface checks, trained latent dynamics code, documentation and synthetic assets.
It does not vendor upstream model code or redistribute upstream neural checkpoints
or benchmark data. KineWorld's small predictor checkpoint is retained in the private
company research release; its separate model card records provenance and limitations.

| Project | Source and license reference | Use here |
|---|---|---|
| DINOv2, Meta | [source](https://github.com/facebookresearch/dinov2), [Apache-2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) | Real pretrained ViT-S/14 inference with frozen weights; feature space for separately trained KineWorld dynamics |
| CausalWM, AetherLabsAI | [source](https://github.com/AetherLabsAI/CausalWM), [LTX-2 Community License](https://github.com/AetherLabsAI/CausalWM/blob/main/LICENSE), [NOTICE](https://github.com/AetherLabsAI/CausalWM/blob/main/NOTICE) | External CoT inference launcher; no weights executed here |
| Wan 2.2, Wan-Video | [source](https://github.com/Wan-Video/Wan2.2), [Apache-2.0](https://github.com/Wan-Video/Wan2.2/blob/main/LICENSE.txt) | External TI2V-5B launcher |
| V-JEPA 2 / 2.1, Meta FAIR | [source](https://github.com/facebookresearch/vjepa2), [MIT](https://github.com/facebookresearch/vjepa2/blob/main/LICENSE), [Apache notice](https://github.com/facebookresearch/vjepa2/blob/main/APACHE-LICENSE) | Optional official encoder/preprocessor via a local checkout |
| Kine-JEPA, KineWorld | [source](https://github.com/kineworld/kine-jepa), [MIT](https://github.com/kineworld/kine-jepa/blob/main/LICENSE) | Optional ActionRollout adapter; trained weights not bundled |
| TriWorldBench | [source](https://github.com/TriWorldBench/TriWorldBench), [data](https://huggingface.co/datasets/TriWorldBench/Dataset) | Input format, external checker and public development examples; no data redistributed |
| OpenCV / NumPy / h5py / FFmpeg | Their upstream distribution terms | Motion estimation, arrays, input reading and video encoding |

The MIT license in this repository applies only to KineWorld-authored files.
Upstream code, model weights, Gemma assets and datasets retain their own terms.
Names and reported upstream scores are not transferred to KineJing. No upstream
partnership or endorsement is implied. No Jev API or proprietary Jev weights are included.

Exact inspected upstream commits are recorded in `kinejing/backends.json`.
