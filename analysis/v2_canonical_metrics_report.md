# Canonical metrics: corrected v2 pipeline

**Model**: Corrected v2 pipeline: chronological block selection (last 2 of 10 = val), no id feature (6 channels only), engineer_features() with inf->0 fix, trained on blocks 0-7 (1,200,000 rows), evaluated on blocks 8-9 (300,000 rows).

**Real Kaggle score (independently confirmed)**: private=0.294, public=0.277  
Submission file: `outputs/v2_corrected_submission.csv` (threshold=0.95)  
Independently confirmed by actual Kaggle submission of this exact corrected model.

**Validation set**: 300,000 rows (blocks [8, 9]), anomaly rate 1.6797%  
**True anomaly events: 6** (lengths, points: [3, 3, 22, 25, 537, 4449])

**Event-count consistency check across all sections**: [6] ✓ consistent

---

## 1–2. Point-wise metrics

| threshold | TN | FP | FN | TP | precision | recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 294,542 | 419 | 4,990 | 49 | 0.104701 | 0.009724 | 0.017796 | 0.596822 | 0.035719 |
| 0.95 | 294,944 | 17 | 5,027 | 12 | 0.413793 | 0.002381 | 0.004736 | 0.596822 | 0.035719 |

(ROC-AUC and PR-AUC are threshold-independent — identical at both rows by construction.)

---

## 3–4. Event-wise metrics (raw and post-processed: min_event_length=3, max_gap=5)

| threshold | type | TPe | FNe | FPe | event recall | event precision (TNR-corrected) | F0.5 |
|---|---|---|---|---|---|---|---|
| 0.5 | raw | 3 | 3 | 383 | 0.500000 | 0.007761 | 0.009664 |
| 0.5 | post-processed | 2 | 4 | 29 | 0.333333 | 0.064481 | 0.076884 |
| 0.95 | raw | 2 | 4 | 16 | 0.333333 | 0.111105 | 0.128198 |
| 0.95 | post-processed | 2 | 4 | 0 | 0.333333 | 1.000000 | 0.714286 |

---

## 5. Size-stratified recall (post-processed), at each threshold

### threshold=0.5

| length (pts) | bin | caught |
|---|---|---|
| 3 | short (<100) | ✗ |
| 3 | short (<100) | ✗ |
| 22 | short (<100) | ✓ |
| 25 | short (<100) | ✓ |
| 537 | medium (100-1000) | ✗ |
| 4449 | long (>1000) | ✗ |

| bin | events | caught | recall | lengths caught | lengths missed |
|---|---|---|---|---|---|
| short (<100) | 4 | 2 | 0.500 | [22, 25] | [3, 3] |
| medium (100-1000) | 1 | 0 | 0.000 | [] | [537] |
| long (>1000) | 1 | 0 | 0.000 | [] | [4449] |

### threshold=0.95

| length (pts) | bin | caught |
|---|---|---|
| 3 | short (<100) | ✗ |
| 3 | short (<100) | ✗ |
| 22 | short (<100) | ✓ |
| 25 | short (<100) | ✓ |
| 537 | medium (100-1000) | ✗ |
| 4449 | long (>1000) | ✗ |

| bin | events | caught | recall | lengths caught | lengths missed |
|---|---|---|---|---|---|
| short (<100) | 4 | 2 | 0.500 | [22, 25] | [3, 3] |
| medium (100-1000) | 1 | 0 | 0.000 | [] | [537] |
| long (>1000) | 1 | 0 | 0.000 | [] | [4449] |

---

## 6. Threshold sweep and bootstrap validation

**Optimal threshold (post-processed event-wise F0.5): 0.95, F0.5=0.714286**

| threshold | TPe | FNe | FPe | recall | precision (corrected) | F0.5 |
|---|---|---|---|---|---|---|
| 0.05 | 6 | 0 | 3076 | 1.0000 | 0.0002 | 0.0003 |
| 0.07 | 6 | 0 | 5917 | 1.0000 | 0.0003 | 0.0004 |
| 0.09 | 5 | 1 | 5521 | 0.8333 | 0.0005 | 0.0006 |
| 0.11 | 5 | 1 | 4786 | 0.8333 | 0.0007 | 0.0009 |
| 0.13 | 4 | 2 | 4091 | 0.6667 | 0.0008 | 0.0009 |
| 0.15 | 3 | 3 | 3246 | 0.5000 | 0.0008 | 0.0010 |
| 0.17 | 3 | 3 | 2316 | 0.5000 | 0.0012 | 0.0015 |
| 0.19 | 3 | 3 | 1614 | 0.5000 | 0.0017 | 0.0022 |
| 0.21 | 3 | 3 | 1105 | 0.5000 | 0.0026 | 0.0032 |
| 0.23 | 3 | 3 | 768 | 0.5000 | 0.0038 | 0.0047 |
| 0.25 | 3 | 3 | 594 | 0.5000 | 0.0049 | 0.0061 |
| 0.27 | 3 | 3 | 480 | 0.5000 | 0.0061 | 0.0076 |
| 0.29 | 3 | 3 | 394 | 0.5000 | 0.0075 | 0.0093 |
| 0.31 | 3 | 3 | 309 | 0.5000 | 0.0095 | 0.0119 |
| 0.33 | 3 | 3 | 248 | 0.5000 | 0.0119 | 0.0148 |
| 0.35 | 3 | 3 | 187 | 0.5000 | 0.0157 | 0.0195 |
| 0.37 | 3 | 3 | 138 | 0.5000 | 0.0212 | 0.0262 |
| 0.39 | 3 | 3 | 105 | 0.5000 | 0.0277 | 0.0342 |
| 0.41 | 3 | 3 | 75 | 0.5000 | 0.0384 | 0.0471 |
| 0.43 | 3 | 3 | 55 | 0.5000 | 0.0517 | 0.0630 |
| 0.45 | 3 | 3 | 41 | 0.5000 | 0.0681 | 0.0824 |
| 0.47 | 3 | 3 | 32 | 0.5000 | 0.0857 | 0.1027 |
| 0.49 | 3 | 3 | 30 | 0.5000 | 0.0909 | 0.1086 |
| 0.51 | 2 | 4 | 28 | 0.3333 | 0.0666 | 0.0793 |
| 0.53 | 2 | 4 | 25 | 0.3333 | 0.0740 | 0.0877 |
| 0.55 | 2 | 4 | 22 | 0.3333 | 0.0833 | 0.0980 |
| 0.57 | 2 | 4 | 21 | 0.3333 | 0.0869 | 0.1020 |
| 0.59 | 2 | 4 | 17 | 0.3333 | 0.1052 | 0.1219 |
| 0.61 | 2 | 4 | 14 | 0.3333 | 0.1250 | 0.1428 |
| 0.63 | 2 | 4 | 11 | 0.3333 | 0.1538 | 0.1724 |
| 0.65 | 2 | 4 | 10 | 0.3333 | 0.1666 | 0.1851 |
| 0.67 | 2 | 4 | 10 | 0.3333 | 0.1666 | 0.1852 |
| 0.69 | 2 | 4 | 8 | 0.3333 | 0.2000 | 0.2174 |
| 0.71 | 2 | 4 | 8 | 0.3333 | 0.2000 | 0.2174 |
| 0.73 | 2 | 4 | 7 | 0.3333 | 0.2222 | 0.2381 |
| 0.75 | 2 | 4 | 6 | 0.3333 | 0.2500 | 0.2631 |
| 0.77 | 2 | 4 | 5 | 0.3333 | 0.2857 | 0.2941 |
| 0.79 | 2 | 4 | 4 | 0.3333 | 0.3333 | 0.3333 |
| 0.81 | 2 | 4 | 3 | 0.3333 | 0.4000 | 0.3846 |
| 0.83 | 2 | 4 | 3 | 0.3333 | 0.4000 | 0.3846 |
| 0.85 | 2 | 4 | 3 | 0.3333 | 0.4000 | 0.3846 |
| 0.87 | 2 | 4 | 3 | 0.3333 | 0.4000 | 0.3846 |
| 0.89 | 2 | 4 | 2 | 0.3333 | 0.5000 | 0.4545 |
| 0.91 | 2 | 4 | 1 | 0.3333 | 0.6667 | 0.5556 |
| 0.93 | 2 | 4 | 1 | 0.3333 | 0.6667 | 0.5556 |
| 0.95 | 2 | 4 | 0 | 0.3333 | 1.0000 | 0.7143  **← optimal** |
| 0.97 | 0 | 6 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 0.99 | 0 | 6 | 0 | 0.0000 | 0.0000 | 0.0000 |

**Bootstrap validation**: 500 iterations, seed=12345, event pool size=6  
Threshold 0.95 was the tied-best choice in **465/500 iterations (93.0% consensus)**  
Best-F0.5-per-iteration: mean=0.5691, median=0.5556, std=0.2417

Top 5 thresholds by bootstrap consensus:

| threshold | iterations tied-best (of 500) |
|---|---|
| 0.95 | 465 |
| 0.49 | 27 |
| 0.11 | 6 |
| 0.13 | 2 |
| 0.05 | 0 |

---

## How to reproduce

```
cd /Users/yaseminates/projects/ESA-Spacecraft-Anomaly-Challenge
python3 analysis/v2_canonical_metrics.py      # recomputes everything above fresh
python3 analysis/generate_report.py           # regenerates this report from the JSON
```

Raw machine-readable values: `analysis/v2_canonical_metrics_results.json`
