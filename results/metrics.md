# Physics Filtering Benchmark

> Supervised demonstration of learning a Z-mass-window (80<M<100 GeV)
> selection from muon kinematics. Labels are a deterministic function of M;
> features already encode M. Not a trigger/discovery benchmark.

**Target Background Acceptance Budget**: 5.0%

**Protocol**: stratified 60/20/20 train/val/test; thresholds fit on
TRAIN (baseline) or VALIDATION (ML) and measured once on frozen TEST.

### Architectures (TEST set, frozen)
| Architecture | Signal Efficiency @5% bg | AUROC | AUPRC | Acc@0.5 |
|---|---|---|---|---|
| Rule-Based Cut (pT_lead >= 23.16 GeV) | **96.71%** | — | — | — |
| Kinematic ML Classifier (XGBoost) | **100.00%** | 0.9981 | 0.9456 | 99.17% |

TEST background retained: baseline 4.99%, ML 4.87% (target 5.0%).

TEST confusion @0.5: TN=18850 FP=115 FN=50 TP=985.

Dummy (all-background) accuracy on TEST would be 94.83%; accuracy alone is misleading under this class imbalance — prefer AUROC/AUPRC above.

> Result: at the same ≈5% background acceptance, ML signal efficiency (100.00%) vs naive pT cut (96.71%): **+3.29pp** difference on frozen TEST.
