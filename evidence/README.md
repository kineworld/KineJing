# Local evidence

Internal exploratory image metrics; not TWB-Score or independent replication.
The fixed 2-pixel correction was checked on additional targets excluded from the example bank.
No causal, statistical-significance or leaderboard-improvement claim is made.

| View | Targets | Median paired PSNR delta (dB) | Median paired SSIM delta |
|---|---:|---:|---:|
| left | 20 | +0.236780 | +0.001155 |
| right | 20 | +0.073520 | +0.000725 |

Head view remains persistence. The global-transform baseline is the paired comparator.
These are pre-encoding frame metrics; compression and the full official evaluator can change conclusions.
Full records: [refinement_confirmation.json](refinement_confirmation.json).

## Rejected experiment

Action-aligned donor retrieval did not show a stable improvement over the revised baseline.
Its paired SSIM medians are listed below; full records also retain the mean regressions.

| View | Median paired SSIM delta |
|---|---:|
| left | +0.000000 |
| right | +0.000000 |

[Rejected retrieval records](rejected_retrieval.json).

## Execution boundaries

- [Local format-check receipt](format_check.json): reports format only, not prediction quality.
- [Synthetic demo manifest](demo_manifest.json): no neural weights or benchmark data.
- External neural checkpoint inference and end-to-end neural composition remain unverified.
- Official KineJing score: unavailable. The research code release does not update a competition entry.
