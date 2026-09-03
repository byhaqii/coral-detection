# Baseline A — Evaluasi Test Set

Top-1 accuracy: **97.98%** (436/445)

## Jumlah data

| Split | Healthy | Bleached | Dead | Total |
|---|---:|---:|---:|---:|
| train | 1378 | 1328 | 406 | 3112 |
| val | 394 | 379 | 116 | 889 |
| test | 197 | 190 | 58 | 445 |

## Confusion matrix

Baris = label sebenarnya; kolom = prediksi model.

| Actual \ Predicted | Healthy | Bleached | Dead |
|---|---:|---:|---:|
| Healthy | 191 | 6 | 0 |
| Bleached | 3 | 187 | 0 |
| Dead | 0 | 0 | 58 |

## Per kelas

| Kelas | Benar/Total | Recall |
|---|---:|---:|
| Healthy | 191/197 | 96.95% |
| Bleached | 187/190 | 98.42% |
| Dead | 58/58 | 100.00% |

Artefak: `confusion_matrix_test.png`, `misclassified_examples.png`, dan training curve `results.png`.
