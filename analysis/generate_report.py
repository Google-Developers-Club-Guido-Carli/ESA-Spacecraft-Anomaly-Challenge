#!/usr/bin/env python3
"""Renders analysis/v2_canonical_metrics_results.json into a human-readable
markdown report (analysis/v2_canonical_metrics_report.md). Pure formatting --
does not recompute anything. Run after v2_canonical_metrics.py."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO / 'analysis'

with open(ANALYSIS_DIR / 'v2_canonical_metrics_results.json') as f:
    r = json.load(f)

lines = []
a = lines.append

a("# Canonical metrics: corrected v2 pipeline")
a("")
a(f"**Model**: {r['model_description']}")
a("")
k = r['real_kaggle_score']
a(f"**Real Kaggle score (independently confirmed)**: private={k['private']}, public={k['public']}  ")
a(f"Submission file: `{k['submission_file']}` (threshold={k['submission_threshold']})  ")
a(f"{k['note']}")
a("")
v = r['validation_set']
a(f"**Validation set**: {v['n_rows']:,} rows (blocks {v['val_block_idx']}), "
  f"anomaly rate {v['anomaly_rate_pct']:.4f}%  ")
a(f"**True anomaly events: {v['n_true_events']}** (lengths, points: {v['true_event_lengths']})")
a("")
cc = r['event_count_consistency_check']
a(f"**Event-count consistency check across all sections**: {cc['values_seen']} "
  f"{'✓ consistent' if cc['consistent'] else '✗ INCONSISTENT — bug'}")
a("")
a("---")
a("")
a("## 1–2. Point-wise metrics")
a("")
a("| threshold | TN | FP | FN | TP | precision | recall | F1 | ROC-AUC | PR-AUC |")
a("|---|---|---|---|---|---|---|---|---|---|")
for thr_key in ["0.5", "0.95"]:
    p = r['point_wise_metrics'][thr_key]
    a(f"| {p['threshold']} | {p['TN']:,} | {p['FP']:,} | {p['FN']:,} | {p['TP']:,} | "
      f"{p['precision']:.6f} | {p['recall']:.6f} | {p['f1']:.6f} | {p['roc_auc']:.6f} | {p['pr_auc']:.6f} |")
a("")
a("(ROC-AUC and PR-AUC are threshold-independent — identical at both rows by construction.)")
a("")
a("---")
a("")
a("## 3–4. Event-wise metrics (raw and post-processed: min_event_length=3, max_gap=5)")
a("")
a("| threshold | type | TPe | FNe | FPe | event recall | event precision (TNR-corrected) | F0.5 |")
a("|---|---|---|---|---|---|---|---|")
for thr_key in ["0.5", "0.95"]:
    ew = r['event_wise_metrics'][thr_key]
    for kind_key, kind_label in [("raw", "raw"), ("post_processed", "post-processed")]:
        e = ew[kind_key]
        a(f"| {thr_key} | {kind_label} | {e['TPe']} | {e['FNe']} | {e['FPe']} | "
          f"{e['event_recall']:.6f} | {e['event_precision_corrected']:.6f} | {e['f05']:.6f} |")
a("")
a("---")
a("")
a("## 5. Size-stratified recall (post-processed), at each threshold")
a("")
for thr_key in ["0.5", "0.95"]:
    sd = r['size_stratified_recall'][thr_key]
    a(f"### threshold={thr_key}")
    a("")
    a("| length (pts) | bin | caught |")
    a("|---|---|---|")
    for ev in sd['events']:
        a(f"| {ev['length']} | {ev['bin']} | {'✓' if ev['caught'] else '✗'} |")
    a("")
    a("| bin | events | caught | recall | lengths caught | lengths missed |")
    a("|---|---|---|---|---|---|")
    for b, s in sd['bins'].items():
        rstr = f"{s['recall']:.3f}" if s['recall'] is not None else "n/a"
        a(f"| {b} | {s['n_events']} | {s['n_caught']} | {rstr} | "
          f"{s['event_lengths_caught']} | {s['event_lengths_missed']} |")
    a("")
a("---")
a("")
a("## 6. Threshold sweep and bootstrap validation")
a("")
ts = r['threshold_sweep']
a(f"**Optimal threshold (post-processed event-wise F0.5): {ts['optimal_threshold']}, "
  f"F0.5={ts['optimal_f05']:.6f}**")
a("")
a("| threshold | TPe | FNe | FPe | recall | precision (corrected) | F0.5 |")
a("|---|---|---|---|---|---|---|")
for row in ts['sweep_table']:
    marker = "  **← optimal**" if abs(row['threshold'] - ts['optimal_threshold']) < 1e-9 else ""
    a(f"| {row['threshold']:.2f} | {row['TPe']} | {row['FNe']} | {row['FPe']} | "
      f"{row['event_recall']:.4f} | {row['event_precision_corrected']:.4f} | {row['f05']:.4f}{marker} |")
a("")
bs = r['bootstrap_validation']
a(f"**Bootstrap validation**: {bs['n_iterations']} iterations, seed={bs['seed']}, "
  f"event pool size={bs['event_pool_size']}  ")
a(f"Threshold {bs['threshold_tested']} was the tied-best choice in "
  f"**{bs['consensus_count']}/{bs['n_iterations']} iterations ({bs['consensus_pct']:.1f}% consensus)**  ")
a(f"Best-F0.5-per-iteration: mean={bs['best_f05_per_iteration_mean']:.4f}, "
  f"median={bs['best_f05_per_iteration_median']:.4f}, std={bs['best_f05_per_iteration_std']:.4f}")
a("")
a("Top 5 thresholds by bootstrap consensus:")
a("")
a("| threshold | iterations tied-best (of 500) |")
a("|---|---|")
for row in bs['top5_thresholds_by_consensus']:
    a(f"| {row['threshold']:.2f} | {row['count']} |")
a("")
a("---")
a("")
a("## How to reproduce")
a("")
a("```")
a("cd /Users/yaseminates/projects/ESA-Spacecraft-Anomaly-Challenge")
a("python3 analysis/v2_canonical_metrics.py      # recomputes everything above fresh")
a("python3 analysis/generate_report.py           # regenerates this report from the JSON")
a("```")
a("")
a("Raw machine-readable values: `analysis/v2_canonical_metrics_results.json`")

with open(ANALYSIS_DIR / 'v2_canonical_metrics_report.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f"Wrote {ANALYSIS_DIR / 'v2_canonical_metrics_report.md'}")
