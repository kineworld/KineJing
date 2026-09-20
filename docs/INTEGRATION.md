# Integration guide

KineJing composes interfaces, not neural weights. Keep a separate Python environment
for each upstream project; set `python` in each example JSON to that environment's
interpreter. Paths are resolved from the directory where the CLI is invoked.
Use `--dry-run` to inspect an argv array; execution never invokes a shell.

## Obtain sources and weights

Read `kinejing/backends.json`, clone the indicated repository into `external/`,
and `git checkout` its exact `revision`. The launcher refuses a different commit.
Install the upstream project's dependencies in its environment. Download the
upstream checkpoints under their own terms into `checkpoints/`; there are no
implicit downloads and no random-weight fallback in these adapters.

CausalWM requires its CoT checkpoint, LTX-2.3 base checkpoint and Gemma text encoder.
The published interface takes a single image and text. Its leaderboard action/multi-view
configuration is not reproduced by this launcher. H200 is the upstream tested hardware;
we have not validated a smaller GPU configuration.

Wan uses the released TI2V-5B image/text path. CPU offloading flags do not establish
compatibility with a particular GPU. Both video launchers explicitly reject an
`actions` field rather than silently ignoring robot controls. Neither produces
synchronized tri-view robot predictions through this interface.

## Video → features

`examples/vjepa21.json` consumes an MP4 and a local
`vjepa2_1_vitb_dist_vitG_384.pt` checkpoint. The worker uses the official preprocessing
and encoder definition, samples 64 frames, loads `ema_encoder` strictly and writes:

- `tokens`: B × N × D floating-point tokens;
- `embedding`: token-mean vector;
- `metadata`: encoder identifier and checkpoint SHA-256.

The pinned upstream hub loader currently points automatic checkpoint downloads at
localhost. This adapter constructs with `pretrained=False` then explicitly loads
the supplied checkpoint, failing on missing or mismatched parameters. It does not
run or score the randomly initialized encoder. This path still requires a real
checkpoint inference validation; its existence is not a model reproduction claim.

## Features + actions → predicted features

The Kine-JEPA adapter loads `ActionRollout` from the pinned first-party repository.
Its checkpoint must contain:

```python
{
    "config": {"dim": 768, "depth": 2, "heads": 12, "action_dim": 14},
    "state_dict": trained_rollout.state_dict(),
    "action_schema": {"columns": ["..."], "units": ["..."],
                      "coordinate_frame": "robot_base", "normalization": "training-transform-v1"},
    "encoder_provenance": {"encoder": "vjepa2_1_vit_base_384", "checkpoint_sha256": "..."}
}
```

The dimensions above illustrate the protocol; choose them to match the actual
encoder and embodiment, and train that predictor before use. No such trained
checkpoint is bundled or claimed. Do not serialize a random model as a trained one.

The input NPZ contains the encoder `tokens`, unchanged `metadata`, `action_schema`,
and an `actions` array B × H × A. Declare every action column and unit, coordinate
frame, and a normalization identifier that refers to the exact training transform.
The full schema must equal the checkpoint's schema; missing schema is rejected.
The packer preserves values and does not normalize them. A legacy checkpoint must
be annotated from its actual training records, never with guessed conventions.

```bash
python -m kinejing pack-actions --features runs/encoded.npz --actions data/actions.npy --schema data/action-schema.json --output data/tokens-and-actions.npz
```

This closes the encoder-to-dynamics input connection but does not supply trained
weights or validate robot prediction quality. The output contains the full `rollout`, final tokens,
mean embedding, source feature metadata and predictor checkpoint digest.
Dimension/provenance mismatches fail before inference. Feature similarity cannot
substitute for a closed-loop robot evaluation.

## Compare futures

`rank` compares candidate NPZ embeddings with an explicitly supplied goal NPZ.
All must use the same encoder and checkpoint. An actual goal image supplied for
planning is acceptable; hidden benchmark future frames are not inference inputs.
This command is a diagnostic cosine ranking, not a learned fusion policy or
official benchmark evaluator.

## CPU path

The `demo` command executes the motion pipeline on a generated geometric scene.
`triworld` runs on official input directories and the separate public example bank.
The CPU baseline has no learned neural weights. Its format validation and local
image metrics are recorded under `evidence/`; it cannot synthesize unseen content.
