# SA-CNFolk Reproduction

This repo contains a PyTorch reproduction entry point for the paper method in
`paper.pdf`, using `songs_dataset.json` and the `.wav` files under `wavs/`.

## What The Script Implements

- Splits by unique `title`, then assigns all renditions of the same title into
  the same subset. With the current metadata this gives 48/6/6 unique titles.
- Extracts log-Mel spectrograms with `sr=44100`, `hop_length=512`,
  `n_mels=128`, `fmax=8000`, matching the author snippet.
- Implements SA-CNFolk:
  `Mel -> CNN embedding -> fold_time aggregation -> Bi-LSTM -> boundary logits`.
- Uses `BCEWithLogitsLoss`, Adam with `lr=1e-3`, `batch_size=1`, local-maximum
  post-processing, validation early stopping, and `ReduceLROnPlateau`.
- Prints precision, recall, and F1 every epoch, and writes them to
  `train_log.csv`.

## Install

On the remote machine, keep your existing CUDA PyTorch install
(`torch==2.10.0+cu128`) and install only the lightweight extras:

```bash
pip install -r requirements.txt
```

## Dry Run

```bash
python3 train_sacnfolk.py --dry-run
```

Expected split summary for the current dataset:

```text
Loaded 548 audio files from 60 unique song titles.
train: 48 titles, 451 files
  val:  6 titles,  37 files
 test:  6 titles,  60 files
```

## Train With Paper-Best Defaults

```bash
python3 train_sacnfolk.py \
  --metadata songs_dataset.json \
  --wav-dir wavs \
  --output-dir runs/sacnfolk_fold1_embed24_h128_l2 \
  --device cuda \
  --batch-size 1 \
  --lr 1e-3 \
  --fold-time 1.0 \
  --dim-embed 24 \
  --lstm-hidden-size 128 \
  --lstm-num-layers 2
```

Useful outputs:

- `runs/.../split_by_title.json`: exact title/file split
- `runs/.../train_log.csv`: per-epoch loss, precision, recall, F1
- `runs/.../best_model.pt`: checkpoint selected by validation F1
- `runs/.../latest_model.pt`: latest checkpoint
- `runs/.../cache/...`: cached log-Mel features

## Main Tunable Parameters

The paper and author logs tune these heavily, so they are command-line args:

- `--fold-time`: feature aggregation window in seconds. Paper candidates:
  `0.25`, `0.5`, `1.0`.
- `--dim-embed`: CNN channels `sigma`. Paper candidates: `6`, `12`, `24`.
- `--lstm-hidden-size`: Bi-LSTM hidden size. Paper candidates:
  `64`, `128`, `256`.
- `--lstm-num-layers`: Bi-LSTM layers. Paper candidates: `1`, `2`, `3`.
- `--label-tolerance-sec`: positive label radius for training labels.
- `--eval-tolerance-sec`: matching tolerance for reported P/R/F. Use `3.0`
  for HR3F and `0.5` for HR.5F.
- `--peak-threshold`, `--peak-filter-size`: local-max post-processing.
- `--metric-average`: `macro` or `micro` reporting/selection.
