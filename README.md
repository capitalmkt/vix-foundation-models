# VIX Foundation-Model Forecasting Benchmark

This repository benchmarks pretrained time-series foundation models against a
classical HAR-OLS baseline on the Cboe Volatility Index (VIX), for both
point/quantile forecasting and three-class direction classification.

Four model families are tested, each in up to three adaptations, on two
information sets:

| Model family | Zero-shot forecasting | Fine-tuned forecasting | Direction classification |
|---|---|---|---|
| Chronos-Bolt | univariate + multivariate | univariate + multivariate | univariate + multivariate |
| Chronos-T5-Large | univariate | univariate | univariate |
| Chronos-2 | univariate + multivariate | univariate + multivariate | univariate + multivariate |
| TimesFM 2.5 | univariate + multivariate | univariate + multivariate | univariate + multivariate |

- **Univariate**: VIX history only.
- **Multivariate**: VIX + VVIX, with VVIX entered as an AR(5) residualized covariate.
- **Benchmark**: HAR-OLS (Corsi's daily/weekly/monthly structure) on log-VIX, evaluated on the same windows and horizons as every model.
- **Fixed protocol throughout**: windows of 16, 32, 64, 128, 256 observations; horizons of 1, 5 and 22 trading days; five fixed seeds where a model is trained; no live data downloads, only an approved local CSV.

This project started as a set of independent Google Colab notebooks (one per
model/adaptation/information-set combination) and was reorganized here into
standalone scripts with a small shared library, so it can be read, run and
extended outside Colab.

## Repository structure

```
.
├── data/                    # place VIX.csv and VVIX.csv here, see data/README.md
├── results/                 # each script writes one CSV table here
│   └── figures/             # created automatically at runtime for PNG plots
├── requirements-chronos.txt # env for the Chronos-Bolt / Chronos-T5-Large / Chronos-2 scripts
├── requirements-timesfm.txt # env for the TimesFM 2.5 scripts
└── src/
    ├── common/              # shared, deduplicated utilities
    │   ├── reproducibility.py  # set_seed
    │   ├── data_io.py          # CSV lookup, results path
    │   ├── evaluation.py       # small metric helpers used by several scripts
    │   └── plotting.py         # forecast/metric plot helpers and PNG export
    ├── univariate/          # 12 scripts, VIX-only information set
    └── multivariate/        # 9 scripts, VIX+VVIX information set
```

Each script under `src/univariate` and `src/multivariate` is standalone and
runnable on its own; every one starts with a docstring describing its exact
model checkpoint, adaptation, fixed windows/horizons and output file.

## Requirements

- A CUDA GPU with at least 16 GB of VRAM (tested on a Colab T4). Scripts check for CUDA at startup and raise an error if none is found.
- Two separate environments, because the Chronos family and the TimesFM 2.5
  family pin different, mutually incompatible `transformers` requirements:

  ```bash
  python -m venv .venv-chronos && source .venv-chronos/bin/activate
  pip install -r requirements-chronos.txt
  deactivate

  python -m venv .venv-timesfm && source .venv-timesfm/bin/activate
  pip install -r requirements-timesfm.txt
  deactivate
  ```

  See the comment at the top of `requirements-timesfm.txt` if pip's resolver
  still reports a conflict on your platform.

## Data

Not included. Place `VIX.csv` and `VVIX.csv` (for the multivariate scripts)
in `data/`. See [`data/README.md`](data/README.md) for accepted schemas and
official sources.

## Running a script

```bash
source .venv-chronos/bin/activate   # or .venv-timesfm, depending on the script
python src/univariate/01_chronos_bolt_zero_shot_forecasting.py
```

The script prints its environment info, runs the fixed evaluation protocol
for every window, shows its plots (and also saves each one as a PNG under
`results/figures/`, since `plt.show()` alone only renders when the code runs
inside a live notebook kernel, not when launched as `!python script.py`),
and writes one CSV to `results/`.

## Design notes

- **No live data**: every script refuses to download data at runtime; it
  only reads the approved local CSV. This keeps every run reproducible
  against a fixed, known input.
- **Deduplicated utilities**: functions that were byte-identical across many
  of the original notebook cells (seeding, CSV lookup, a couple of small
  metric helpers) now live once in `src/common/` and are
  imported everywhere instead of being copy-pasted. Functions that differed
  in meaningful ways between scripts, even subtly, were deliberately left
  as they were rather than force-merged.
- **Pinned checkpoints**: every model is pinned to an exact revision hash,
  not just a model name, so a run today loads the same weights as the
  original experiments.

## Known limitations

- While the scripts are now supported locally, a few in-script messages retain legacy Colab references (such as 'T4 GPU runtime in Google Colab'). These were left unchanged as they remain functionally accurate given that a CUDA-compatible GPU is still required.
  The GPU check is enforced in Python; it does not depend on Colab.
- This reorganization was validated by splitting the original notebook
  export, deduplicating identical functions and checking that every resulting file compiles.
