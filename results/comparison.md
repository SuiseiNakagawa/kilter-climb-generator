| Model | Split | MAE (points) | MAE (V-grades) | Exact V | Within ±1 V |
|---|---|---|---|---|---|
| Angle only (baseline) | val | 3.09 | 1.92 | 17.6% | 46.2% |
| Angle only (baseline) | test | 3.15 | 1.95 | 16.8% | 44.9% |
| Boosting: geometry | val | 1.82 | 1.15 | 29.6% | 68.8% |
| Boosting: geometry | test | 1.83 | 1.16 | 29.5% | 69.0% |
| Boosting: geometry + holds | val | 1.39 | 0.89 | 36.3% | 79.8% |
| Boosting: geometry + holds | test | 1.40 | 0.89 | 36.6% | 80.1% |
| Graph neural network | val | 1.33 | 0.84 | 39.0% | 81.9% |
| Graph neural network | test | 1.34 | 0.84 | 38.5% | 82.2% |
