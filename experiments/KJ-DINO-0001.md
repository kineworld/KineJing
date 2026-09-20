# KJ-DINO-0001: pretrained vision + learned three-camera dynamics

Preregistered before feature extraction/training. Seed 20260920.

Use official DINOv2 ViT-S/14 frozen weights, not a randomly initialized encoder.
For each of 100 public TriWorldBench validation episodes, sample nine uniformly
spaced frames including the initial frame. Concatenate CLS and mean patch features
for each of the three cameras. Joint actions retain the dataset's 14-column order;
no claim about robot units or compatibility with other robots is made.

Split by episode, not frames: IDs divisible by ten are final local evaluation;
IDs ending in nine are checkpoint selection; remaining 80 are training.
These public examples have previously been explored for CPU baselines, so this is
not a blind benchmark or the official test set. No official test future is used.

Fit train-only feature/action statistics. Train a shared initial-vision projection,
an action-prefix GRU and a residual latent prediction head. Select checkpoint by
validation latent MSE over 120 epochs. Evaluate final local split once against
initial-frame persistence, and shuffle whole action trajectories across episodes
as a sensitivity diagnostic. Report per-episode and per-camera latent error.

Do not tune architecture after inspecting final local evaluation. Keep negative
results. A lower DINO latent error is not evidence of improved RGB, physical
accuracy, causal identification, robot control or TWB-Score. There is no RGB decoder
or language conditioning in this experiment. This is a small research predictor,
not a newly pretrained foundation model or a merge of incompatible model weights.
