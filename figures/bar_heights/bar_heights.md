# Bar Heights for September 2026 Figures

This document tabulates the exact bar heights (success rates in %) and bootstrap 95% confidence intervals for all bar plot figures in `figures/sept2026_version/` where number labels were removed from the plots.

**Note on Overlay Stacking:** In all figures, bars are overlaid (Acceptable encompasses Medium, which encompasses High). The bar heights listed below represent the absolute success rate (%) for each quality tier.

## Figure 1: CAPRI Score Set (n=69)

*Point estimates computed via `M.success_at_k`.*

| Evaluation Metric | Model | Acceptable (%) | Medium (%) | High (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Top-1** | Baseline DFMDock | 30.4% | 17.4% | 7.2% |
| **Top-1** | LambdaDockScore | 31.9% | 26.1% | 1.4% |
| **Top-1** | EuDockScore | 20.3% | 13.0% | 0.0% |
| **Top-5** | Baseline DFMDock | 44.9% | 31.9% | 7.2% |
| **Top-5** | LambdaDockScore | 49.3% | 34.8% | 7.2% |
| **Top-5** | EuDockScore | 40.6% | 26.1% | 4.3% |

---

## Figure 2: DB5.5 (n=253)

*Mean performance from 10,000 pose-bootstrap samples. 95% confidence intervals reported for all three quality tiers (represented as error bars on the plots).*

| Evaluation Metric | Model | Acceptable (%) [95% CI] | Medium (%) [95% CI] | High (%) [95% CI] |
| :--- | :--- | :---: | :---: | :---: |
| **Top-1** | Baseline DFMDock | 3.4% [2.4, 4.3] | 0.8% [0.4, 1.6] | 0.0% [0.0, 0.0] |
| **Top-1** | LambdaDockScore | 5.0% [3.6, 6.3] | 1.3% [0.8, 2.0] | 0.3% [0.0, 0.4] |
| **Top-5** | Baseline DFMDock | 7.2% [5.5, 8.7] | 2.0% [1.2, 2.8] | 0.2% [0.0, 0.4] |
| **Top-5** | LambdaDockScore | 9.4% [7.9, 11.1] | 2.3% [1.6, 3.2] | 0.3% [0.0, 0.4] |
| **Oracle** | Baseline DFMDock | 31.1% [28.9, 33.2] | 6.1% [4.7, 7.1] | 0.3% [0.0, 0.4] |
| **Oracle** | LambdaDockScore | 31.1% [28.9, 33.2] | 6.1% [4.7, 7.1] | 0.3% [0.0, 0.4] |

---

## Figure 3: DB5.5 Antibody-Antigen Complexes (n=55)

*Mean performance from 10,000 pose-bootstrap samples (subset: `cat == 'AA'`). 95% confidence intervals reported for all three quality tiers.*

| Evaluation Metric | Model | Acceptable (%) [95% CI] | Medium (%) [95% CI] | High (%) [95% CI] |
| :--- | :--- | :---: | :---: | :---: |
| **Top-1** | Baseline DFMDock | 1.2% [0.0, 1.8] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| **Top-1** | LambdaDockScore | 1.9% [0.0, 3.6] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| **Top-5** | Baseline DFMDock | 2.5% [0.0, 5.5] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| **Top-5** | LambdaDockScore | 5.1% [1.8, 7.3] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| **Oracle** | Baseline DFMDock | 17.6% [14.5, 20.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| **Oracle** | LambdaDockScore | 17.6% [14.5, 20.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |

---

## Figure S2: DB5.5 Split into 5 ΔASA Bins

*Mean performance from 10,000 pose-bootstrap samples across 5 interface area bins. 95% confidence intervals reported for all three quality tiers.*

### (a) 0-20th pct (799-1316 Å², n=51)

| Evaluation Metric | Model | Acceptable (%) [95% CI] | Medium (%) [95% CI] | High (%) [95% CI] |
| :--- | :--- | :---: | :---: | :---: |
| **Top-1** | Baseline DFMDock | 0.2% [0.0, 2.0] | 0.2% [0.0, 2.0] | 0.0% [0.0, 0.0] |
| **Top-1** | LambdaDockScore | 2.1% [0.0, 5.9] | 0.2% [0.0, 2.0] | 0.0% [0.0, 0.0] |
| **Top-5** | Baseline DFMDock | 1.9% [0.0, 3.9] | 1.2% [0.0, 2.0] | 0.0% [0.0, 0.0] |
| **Top-5** | LambdaDockScore | 7.6% [3.9, 11.8] | 1.7% [0.0, 3.9] | 0.0% [0.0, 0.0] |
| **Oracle** | Baseline DFMDock | 31.5% [25.5, 37.3] | 5.0% [2.0, 7.8] | 0.0% [0.0, 0.0] |
| **Oracle** | LambdaDockScore | 31.5% [25.5, 37.3] | 5.0% [2.0, 7.8] | 0.0% [0.0, 0.0] |

### (b) 20-40th pct (1317-1617 Å², n=50)

| Evaluation Metric | Model | Acceptable (%) [95% CI] | Medium (%) [95% CI] | High (%) [95% CI] |
| :--- | :--- | :---: | :---: | :---: |
| **Top-1** | Baseline DFMDock | 1.3% [0.0, 4.0] | 0.2% [0.0, 2.0] | 0.0% [0.0, 0.0] |
| **Top-1** | LambdaDockScore | 10.0% [6.0, 14.0] | 3.8% [2.0, 6.0] | 1.3% [0.0, 2.0] |
| **Top-5** | Baseline DFMDock | 6.1% [4.0, 10.0] | 2.6% [0.0, 4.0] | 0.8% [0.0, 2.0] |
| **Top-5** | LambdaDockScore | 14.1% [10.0, 18.0] | 5.2% [4.0, 6.0] | 1.3% [0.0, 2.0] |
| **Oracle** | Baseline DFMDock | 33.3% [28.0, 38.0] | 9.0% [6.0, 10.0] | 1.3% [0.0, 2.0] |
| **Oracle** | LambdaDockScore | 33.3% [28.0, 38.0] | 9.0% [6.0, 10.0] | 1.3% [0.0, 2.0] |

### (c) 40-60th pct (1619-1866 Å², n=51)

| Evaluation Metric | Model | Acceptable (%) [95% CI] | Medium (%) [95% CI] | High (%) [95% CI] |
| :--- | :--- | :---: | :---: | :---: |
| **Top-1** | Baseline DFMDock | 8.2% [5.9, 9.8] | 3.2% [2.0, 3.9] | 0.0% [0.0, 0.0] |
| **Top-1** | LambdaDockScore | 3.7% [0.0, 7.8] | 1.5% [0.0, 2.0] | 0.0% [0.0, 0.0] |
| **Top-5** | Baseline DFMDock | 9.1% [5.9, 11.8] | 4.6% [2.0, 5.9] | 0.0% [0.0, 0.0] |
| **Top-5** | LambdaDockScore | 7.1% [3.9, 9.8] | 2.0% [2.0, 2.0] | 0.0% [0.0, 0.0] |
| **Oracle** | Baseline DFMDock | 26.3% [21.6, 29.4] | 7.7% [5.9, 7.8] | 0.0% [0.0, 0.0] |
| **Oracle** | LambdaDockScore | 26.3% [21.6, 29.4] | 7.7% [5.9, 7.8] | 0.0% [0.0, 0.0] |

### (d) 60-80th pct (1871-2270 Å², n=50)

| Evaluation Metric | Model | Acceptable (%) [95% CI] | Medium (%) [95% CI] | High (%) [95% CI] |
| :--- | :--- | :---: | :---: | :---: |
| **Top-1** | Baseline DFMDock | 2.6% [0.0, 6.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| **Top-1** | LambdaDockScore | 1.8% [0.0, 4.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| **Top-5** | Baseline DFMDock | 7.0% [4.0, 10.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| **Top-5** | LambdaDockScore | 3.8% [2.0, 4.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| **Oracle** | Baseline DFMDock | 25.9% [20.0, 32.0] | 2.5% [0.0, 4.0] | 0.0% [0.0, 0.0] |
| **Oracle** | LambdaDockScore | 25.9% [20.0, 32.0] | 2.5% [0.0, 4.0] | 0.0% [0.0, 0.0] |

### (e) 80-100th pct (2278-6671 Å², n=51)

| Evaluation Metric | Model | Acceptable (%) [95% CI] | Medium (%) [95% CI] | High (%) [95% CI] |
| :--- | :--- | :---: | :---: | :---: |
| **Top-1** | Baseline DFMDock | 4.5% [2.0, 7.8] | 0.5% [0.0, 2.0] | 0.0% [0.0, 0.0] |
| **Top-1** | LambdaDockScore | 7.5% [3.9, 11.8] | 0.9% [0.0, 3.9] | 0.0% [0.0, 0.0] |
| **Top-5** | Baseline DFMDock | 11.9% [7.8, 15.7] | 1.5% [0.0, 3.9] | 0.0% [0.0, 0.0] |
| **Top-5** | LambdaDockScore | 14.5% [9.8, 19.6] | 2.7% [0.0, 5.9] | 0.0% [0.0, 0.0] |
| **Oracle** | Baseline DFMDock | 38.4% [33.3, 41.2] | 6.2% [3.9, 7.8] | 0.0% [0.0, 0.0] |
| **Oracle** | LambdaDockScore | 38.4% [33.3, 41.2] | 6.2% [3.9, 7.8] | 0.0% [0.0, 0.0] |

