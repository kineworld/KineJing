"""Generate/check prose numbers from committed paired-result artifacts."""
import argparse,json,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def document():
    report=json.loads((ROOT/'evidence/refinement_confirmation.json').read_text())
    rejected=json.loads((ROOT/'evidence/rejected_retrieval.json').read_text())
    rows=report['records'];held=set(report['heldout']);bank=set(report['bank_ids'])
    assert len(rows)==len(held)==20 and not held&bank
    assert {r['episode'] for r in rows}==held
    out=['# Local evidence','',
         'Internal exploratory image metrics; not TWB-Score or independent replication.',
         'The fixed 2-pixel correction was checked on additional targets excluded from the example bank.',
         'No causal, statistical-significance or leaderboard-improvement claim is made.','',
         '| View | Targets | Median paired PSNR delta (dB) | Median paired SSIM delta |',
         '|---|---:|---:|---:|']
    for view in ['left','right']:
        values={m:statistics.median([r['views'][view]['bounded_2px'][m]-r['views'][view]['baseline'][m] for r in rows]) for m in ['psnr','ssim']}
        for m in values:assert abs(values[m]-report['summary'][view]['bounded_2px'][m]['median_delta'])<1e-10
        out.append('| {} | {} | {:+.6f} | {:+.6f} |'.format(view,len(rows),values['psnr'],values['ssim']))
    out += ['','Head view remains persistence. The global-transform baseline is the paired comparator.',
            'These are pre-encoding frame metrics; compression and the full official evaluator can change conclusions.',
            'Full records: [refinement_confirmation.json](refinement_confirmation.json).','',
            '## Rejected experiment','',
            'Action-aligned donor retrieval did not show a stable improvement over the revised baseline.',
            'Its paired SSIM medians are listed below; full records also retain the mean regressions.','',
            '| View | Median paired SSIM delta |','|---|---:|']
    for view in ['left','right']:
        vals=[r['views'][view]['aligned']['ssim']-r['views'][view]['revision2']['ssim'] for r in rejected['records']]
        out.append('| {} | {:+.6f} |'.format(view,statistics.median(vals)))
    out += ['','[Rejected retrieval records](rejected_retrieval.json).','',
            '## Execution boundaries','',
            '- [Local format-check receipt](format_check.json): reports format only, not prediction quality.',
            '- [Synthetic demo manifest](demo_manifest.json): no neural weights or benchmark data.',
            '- DINOv2 pretrained inference and a small learned latent predictor are recorded in [the model card](../docs/DYNAMICS_MODEL_CARD.md); large generative models and end-to-end composition remain unverified.',
            '- Official KineJing score: unavailable. The research code release does not update a competition entry.','']
    return '\n'.join(out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    expected=document();path=ROOT/'evidence/README.md'
    if a.write:path.write_text(expected,encoding='utf-8')
    assert path.read_text(encoding='utf-8')==expected,'Evidence prose differs from artifacts'
    print('Evidence claims match committed paired results.')

if __name__=='__main__':main()
