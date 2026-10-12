# BeatGraphBP

Cuffless blood pressure (SBP/DBP) estimation from ECG and PPG using a 1D U-Net combined with graph attention over detected cardiac beats. Trained and evaluated on the VitalDB subsets of [PulseDB](https://github.com/pulselabteam/PulseDB).

## How it works

Each 10 s window (1,248 samples at 125 Hz) goes through two paths:

- **Window path:** a 4-level 1D U-Net (32/64/128/256 channels, SE blocks, attention-gated skips) over `[ECG, PPG, PPG', PPG'']`.
- **Beat path:** R-peaks split the window into beats. Each beat is resized to 128 samples and encoded by a small CNN, then four per-beat features are added (RR, delta-RR, normalized HR, approximate PTT). Beats become nodes in a graph with self, temporal, and morphology-similarity edges, followed by two 4-head graph-attention layers.

The beat tokens are attention-pooled into one graph vector and also spread back over their sample positions as context for the U-Net bottleneck. The BP head takes encoder/decoder pooled features, the graph vector, and an embedding of 5 demographics (age, sex, height, weight, BMI) and predicts SBP and DBP. An auxiliary 1x1 conv head reconstructs the ABP waveform during training.

About 1.37M trainable parameters.

```
window ─► U-Net encoder ─► bottleneck ─► fuse ─► U-Net decoder ─► ABP waveform (aux)
   │                                       ▲
   │                                 graph context
   │                                       │
   └─► R-peaks ─► beats ─► BeatEncoder ─► graph attention ─► graph pool
                                                              │
   [encoder pool, decoder mean/max, graph pool, demographics] ─► BP head ─► SBP, DBP
```

## Repository layout

```
config.py       constants: sampling rate, filters, training defaults, cache version
paths.py        data / cache / checkpoint locations
preprocess.py   filtering, R-peak detection, beat boundaries, RR/PTT features
dataset.py      PulseDB .mat reader, cached memmap dataset, split helpers
model.py        BPMGraphUNet
metrics.py      MAE / ME / SD, numerical AAMI check, subject-level aggregation
train.py        training loop
evaluate.py     evaluation with optional calibration
```

## Setup

Python 3.10+ and PyTorch 2.3+.

```bash
pip install torch numpy scipy h5py
```

The modules use relative imports, so the repo has to be imported as a package. Clone it into a folder with a valid Python name (no hyphens):

```bash
git clone https://github.com/darrinbright/beatgraph-cuffless-bp.git beatgraph
```

## Data

Generate the VitalDB subsets with the PulseDB scripts. `paths.py` expects this layout:

```
workspace/
├── Datasets/
│   └── VitalDB/
│       ├── VitalDB_Train_Subset.mat
│       ├── VitalDB_CalBased_Test_Subset.mat
│       ├── VitalDB_CalFree_Test_Subset.mat
│       ├── VitalDB_AAMI_Test_Subset.mat
│       └── VitalDB_AAMI_Cal_Subset.mat
└── project/
    ├── beatgraph/          <- this repo
    ├── checkpoints/        created on first run
    └── results/cache/      created on first run
```

Signal channels are read as ECG = 0, PPG = 1, ABP = 2 (set in `config.py`). If your files are somewhere else, edit `paths.py` or pass `--file` explicitly.

On first use, each subset is preprocessed into a memmap cache under `results/cache/`. The cache key includes the file name, input mode, `CACHE_VERSION`, the selected record indices, and the normalization stats. Bump `CACHE_VERSION` in `config.py` after changing any preprocessing.

## Training

Run from `project/`:

```bash
python -m beatgraph.train \
    --input-mode ecg_ppg_d1_d2 \
    --importance-weighted-sampler \
    --cache-workers 8 \
    --output-dir checkpoints/run1
```

Input modes:

| Mode | Window channels | Beat encoder input |
|---|---|---|
| `ecg_ppg_d1_d2` (default) | ECG, PPG, PPG', PPG'' | ECG, PPG |
| `ppg_d1_d2` | PPG, PPG', PPG'' | PPG, PPG', PPG'' |

Beat boundaries and RR/PTT features are always derived from the ECG channel, so the PPG-only mode still needs ECG present in the `.mat` file.

Defaults: Adam (lr 5e-5, weight decay 1e-4), batch 256, 300 epochs, cosine warm restarts every 50 epochs down to `--lr-min`, AMP on CUDA, gradient clipping at 0.5, L1 loss on standardized SBP/DBP plus L1 loss on the normalized ABP waveform (both weighted 1.0). Early stopping only starts counting after epoch 100.

Useful flags:

| Flag | Effect |
|---|---|
| `--importance-weighted-sampler` | reweights training windows so their SBP distribution matches `--aami-cal-file` |
| `--balanced-sampler` | equalizes sampling across 10 SBP bins |
| `--max-records N` | use only the first N records (quick debugging) |
| `--val-split 0` | disable validation; only `*_last.pt` is saved |
| `--wave-loss-weight`, `--scalar-loss-weight` | loss weights |
| `--accumulate-steps` | gradient accumulation |
| `--resume PATH` | resume from a `*_last.pt` checkpoint |

**Validation split:** 20% of the training file's records are held out at random as a record-level split, so the same subjects appear in both train and validation. The best checkpoint is selected by window-level mean MAE on this split. `--val-split` only switches validation on or off; the 20% fraction is fixed in `build_subject_protocol`.

Outputs in `--output-dir`:
- `bpm_graph_unet_<mode>_best.pt` (best validation MAE)
- `bpm_graph_unet_<mode>_last.pt` (written every epoch)

Checkpoints store the model, optimizer, scheduler, and scaler states, plus the train-set target, demographic, and channel normalization stats that evaluation reuses.

## Evaluation

```bash
# calibration-based test (subjects seen in training)
python -m beatgraph.evaluate \
    --checkpoint checkpoints/run1/bpm_graph_unet_ecg_ppg_d1_d2_best.pt \
    --file ../Datasets/VitalDB/VitalDB_CalBased_Test_Subset.mat \
    --output results/eval/calbased.npz

# calibration-free test (unseen subjects)
python -m beatgraph.evaluate \
    --checkpoint checkpoints/run1/bpm_graph_unet_ecg_ppg_d1_d2_best.pt \
    --file ../Datasets/VitalDB/VitalDB_CalFree_Test_Subset.mat \
    --output results/eval/calfree.npz
```

Reported metrics are MAE, mean error (ME), and SD of the signed error for SBP and DBP, at two levels:
- **Window level:** per window.
- **Subject level:** median prediction vs median target across each subject's windows.

`aami_pass` is only a numerical check (|ME| ≤ 5 and SD ≤ 8 mmHg for both SBP and DBP). It is not a clinical AAMI/ESH/ISO validation.

The `.npz` output contains window and subject predictions, targets, and the metric dicts.
