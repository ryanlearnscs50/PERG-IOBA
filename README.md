# PERG-IOBA: classifying eye disease from PERG waveforms

**Start here → [docs/mac_vs_optic_summary.md](docs/mac_vs_optic_summary.md)**: can the PERG tell macular disease
from optic-nerve disease? A simple 1-D CNN (ResNet) reaches **74% accuracy / AUROC 0.76** vs **63% / 0.58** for a
PCA + logistic-regression baseline (10 repeats of 5-fold cross-validation, grouped by person).

## Data
PERG-IOBA dataset from PhysioNet (https://physionet.org/content/perg-ioba-dataset/1.0.0/), ODC-By licence.
Not included in this repo: download it and place the CSV files in `data/csv/` (`NNNN.csv` + `participants_info.csv`).

## Layout
| Path | What |
|---|---|
| `docs/` | Write-up, figures, and the 10-repeat result table for macular vs optic |
| `src/data.py` | Loading, disease groupings, task definitions (`mac_opt` = macular vs optic) |
| `src/train_dl.py` | CNN training (ResNet and other architectures), person-grouped CV, saves out-of-fold predictions |
| `src/exp_screen.py` | Baseline (PCA + logistic regression) and other classical models, same CV splits |
| `src/w6_score.py`, `src/w6_figs.py` | Score the models over 10 repeats and draw the figures in `docs/` |
| `src/analysis_diag.py` | Per-diagnosis breakdown |
| `scripts/` | Slurm job scripts used to run the experiments |
| other `src/` files | Earlier exploration (TCN, self-supervised pretraining, foundation-model embeddings, leakage checks) |

## Reproduce the headline result
```bash
pip install -r requirements.txt
cd src
# CNN: one CV repeat (seed); repeat for seeds 0-9
python train_dl.py --task mac_opt --arch resnet --mode direct --epochs 300 --seed 0 --name dl_resnet_direct_e300
# Baseline, seeds 0-9
PERG_SEEDS="0 1 2 3 4 5 6 7 8 9" python exp_screen.py mac_opt pca_base
python w6_score.py pca_base dl_resnet_direct_e300
python w6_figs.py dl_resnet_direct_e300
```
Results are written to `runs/screen/mac_opt/<model>/` (not tracked).
