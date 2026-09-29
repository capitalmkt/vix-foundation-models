"""
01. Chronos-Bolt - Zero-Shot Forecasting (univariate)

Model family: Chronos-Bolt
Adaptation: Zero-Shot Forecasting
Information set: VIX univariate only
Colab layout: run this complete file as one cell on a T4 GPU
Fixed windows: 16, 32, 64, 128, 256. No window selection is performed.
Fixed horizons: 1, 5, 22 trading days.
Model ID: amazon/chronos-bolt-base
Checkpoint revision: 5d9f166d69f47aef3401367a7b842e78fe97b121
Forecast configuration: direct 22-step log-VIX forecast, q10/q50/q90, q50 point forecast.
Evaluation configuration: level MAE primary, HAR-OLS benchmark, non-overlapping offset 0 primary.
Output configuration: one compact 15-row table is displayed and saved unchanged to this cell's CSV.
HAR note: Corsi's daily/weekly/monthly structure is estimated by OLS on log-VIX with direct horizons.
Update configuration: No parameter updates. The pretrained pipeline is loaded once and reused for every window.
Dependency policy: the Colab NumPy, pandas, SciPy, statsmodels, scikit-learn and PyTorch stack is preserved.
Live data policy: disabled. An approved VIX CSV is mandatory.

Documentation:
  https://github.com/amazon-science/chronos-forecasting
  https://github.com/amazon-science/chronos-forecasting/blob/main/pyproject.toml
  https://huggingface.co/amazon/chronos-bolt-base
  https://huggingface.co/amazon/chronos-bolt-base/commit/5d9f166d69f47aef3401367a7b842e78fe97b121

Installation notes:
  pip install "chronos-forecasting==2.3.1"

Part of the VIX foundation-model forecasting benchmark.
Run as a standalone script on a CUDA GPU (tested on a T4).
"""

# Imports
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.reproducibility import set_seed
from common.data_io import REPO_DATA_DIR, locate_approved_csv, resolve_results_directory
from common.plotting import enable_figure_autosave
from common.evaluation import mean_absolute_error
import hashlib
import importlib.metadata
import math
import platform
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from IPython.display import display
from scipy.stats import t as student_t

# Configuration
MODEL_FAMILY = "chronos_bolt"
MODEL_ID = "amazon/chronos-bolt-base"
MODEL_REVISION = "5d9f166d69f47aef3401367a7b842e78fe97b121"
WINDOW_SIZES = [16, 32, 64, 128, 256]
HORIZONS = [1, 5, 22]
HORIZON_STEPS = [0, 4, 21]
MAX_HORIZON = 22
MAX_WINDOW = 256
HAR_WEEK = 5
HAR_MONTH = 22
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
INFERENCE_BATCH_SIZE = 128
SEED = 2026
BOOTSTRAP_REPLICATIONS = 5000
OUTPUT_FILENAME = "forecast_chronos_bolt_zero_shot_univariate.csv"
LOCAL_VIX_PATHS = [REPO_DATA_DIR / "VIX.csv", Path("/content/VIX.csv"), Path("/content/VIXCLS.csv"), Path("/content/vix.csv")]
RESULTS_DIRECTORY = resolve_results_directory()
enable_figure_autosave(RESULTS_DIRECTORY, Path(OUTPUT_FILENAME).stem)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

if DEVICE.type != "cuda":
    raise RuntimeError("Select a T4 GPU runtime in Google Colab before running this cell.")
print(
    f"Python {platform.python_version()} | PyTorch {torch.__version__} | "
    f"chronos-forecasting {importlib.metadata.version('chronos-forecasting')} | GPU {torch.cuda.get_device_name(0)}"
)


# Reproducibility


set_seed(SEED)


# Data




def load_vix_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    for date_column, value_column in (("Date", "Close"), ("DATE", "VIXCLS")):
        if {date_column, value_column}.issubset(raw.columns):
            break
    else:
        raise ValueError("Accepted schemas are exactly Date,Close or DATE,VIXCLS.")

    clean = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw[date_column], errors="coerce"),
            "VIX": pd.to_numeric(raw[value_column], errors="coerce"),
        }
    )
    valid = clean["Date"].notna() & clean["VIX"].notna() & clean["VIX"].gt(0)
    dropped_rows = int((~valid).sum())
    clean = clean.loc[valid].sort_values("Date").reset_index(drop=True)
    if clean.empty:
        raise ValueError("The approved CSV contains no valid positive VIX observations.")
    duplicate_dates = clean.loc[clean["Date"].duplicated(keep=False), "Date"]
    if not duplicate_dates.empty:
        raise ValueError(f"Duplicate VIX dates found: {duplicate_dates.head().tolist()}")

    print(f"VIX source: {path}")
    print(f"VIX SHA-256: {hashlib.sha256(path.read_bytes()).hexdigest()}")
    print(f"Dropped rows: {dropped_rows}")
    print(f"Range: {clean['Date'].iloc[0].date()} to {clean['Date'].iloc[-1].date()} | n={len(clean)}")
    return clean


vix_data = load_vix_csv(locate_approved_csv(LOCAL_VIX_PATHS, "VIX"))
dates = vix_data["Date"].to_numpy()
vix_level = vix_data["VIX"].to_numpy(dtype=np.float64)
log_vix = np.log(vix_level)
n_obs = len(vix_data)


# Split
validation_start = int(np.floor(0.60 * n_obs))
test_start = int(np.floor(0.80 * n_obs))
minimum_origin = max(MAX_WINDOW - 1, HAR_MONTH - 1)
validation_origins = np.arange(max(validation_start, minimum_origin), test_start - MAX_HORIZON, dtype=int)
test_origins = np.arange(max(test_start, minimum_origin), n_obs - MAX_HORIZON, dtype=int)

if min(len(validation_origins), len(test_origins)) == 0:
    raise ValueError("The cleaned VIX history is too short for the fixed 60/20/20 protocol.")
print(f"Validation origins: {len(validation_origins)} | Test origins: {len(test_origins)}")


# Contexts
def build_contexts(values: np.ndarray, origins: np.ndarray, window_size: int) -> np.ndarray:
    offsets = np.arange(1 - window_size, 1)
    return values[origins[:, None] + offsets].astype(np.float32, copy=False)


def target_matrix(values: np.ndarray, origins: np.ndarray) -> np.ndarray:
    return np.column_stack([values[origins + horizon] for horizon in HORIZONS])


def safe_exp(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, -20.0, 20.0)
    clip_count = int(np.count_nonzero(clipped != values))
    if clip_count:
        warnings.warn(f"Clipped {clip_count} log forecasts to [-20, 20] before exponentiation.", RuntimeWarning)
    return np.exp(clipped)


def check_quantiles(quantiles: np.ndarray, expected_rows: int) -> np.ndarray:
    quantiles = np.asarray(quantiles, dtype=np.float64)
    expected_shape = (expected_rows, MAX_HORIZON, len(QUANTILE_LEVELS))
    if quantiles.shape != expected_shape:
        raise ValueError(f"Unexpected quantile shape {quantiles.shape}; expected {expected_shape}.")
    if not np.isfinite(quantiles).all():
        raise FloatingPointError("The model returned NaN or infinite quantiles.")
    crossing_count = int(np.count_nonzero(np.diff(quantiles, axis=-1) < 0))
    if crossing_count:
        warnings.warn(f"Detected {crossing_count} quantile crossings; applying monotone rearrangement.", RuntimeWarning)
        quantiles = np.sort(quantiles, axis=-1)
    return quantiles


# HAR benchmark
def har_features(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return np.column_stack(
        [
            np.ones(len(values)),
            values,
            series.rolling(HAR_WEEK, min_periods=HAR_WEEK).mean(),
            series.rolling(HAR_MONTH, min_periods=HAR_MONTH).mean(),
        ]
    )


def expanding_har_ols(values: np.ndarray, origins: np.ndarray) -> dict[int, np.ndarray]:
    features = har_features(values)
    forecasts = {}
    for horizon in HORIZONS:
        log_forecasts = []
        for origin in origins:
            training = np.arange(HAR_MONTH - 1, origin - horizon + 1)
            x_train = features[training]
            y_train = values[training + horizon]
            finite = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)
            if finite.sum() < 60:
                raise ValueError(f"Only {finite.sum()} valid HAR rows are available at origin {origin}.")
            params = sm.OLS(y_train[finite], x_train[finite]).fit().params
            if not np.isfinite(params).all():
                raise FloatingPointError("HAR-OLS returned non-finite coefficients.")
            log_forecasts.append(float(features[origin] @ params))
        forecasts[horizon] = safe_exp(log_forecasts)
    return forecasts


test_har = expanding_har_ols(log_vix, test_origins)


# Statistical comparison


def dm_p_value(loss_differential: np.ndarray) -> float:
    differential = np.asarray(loss_differential, dtype=np.float64)
    if differential.ndim != 1 or not np.isfinite(differential).all():
        raise ValueError("DM requires a finite one-dimensional loss differential.")
    n = len(differential)
    if n < 3:
        return np.nan

    lag = min(int(np.floor(n ** (1 / 3))), n - 1)
    centered = differential - differential.mean()
    long_run_variance = float(centered @ centered / n)
    for current_lag in range(1, lag + 1):
        covariance = float(centered[current_lag:] @ centered[:-current_lag] / n)
        long_run_variance += 2.0 * (1.0 - current_lag / (lag + 1.0)) * covariance
    if long_run_variance <= np.finfo(float).eps:
        return np.nan

    statistic = differential.mean() / math.sqrt(long_run_variance / n)
    statistic *= math.sqrt((n - 1.0) / n)
    return float(2.0 * student_t.sf(abs(statistic), df=n - 1))


def bootstrap_confidence_interval(loss_differential: np.ndarray) -> tuple[float, float]:
    differential = np.asarray(loss_differential, dtype=np.float64)
    if differential.ndim != 1 or not np.isfinite(differential).all():
        raise ValueError("Bootstrap requires a finite one-dimensional loss differential.")
    n = len(differential)
    if n < 2:
        return np.nan, np.nan

    block_length = min(max(2, round(n ** (1 / 3))), n)
    starts = np.arange(n - block_length + 1)
    blocks_needed = math.ceil(n / block_length)
    rng = np.random.default_rng(SEED)
    means = np.empty(BOOTSTRAP_REPLICATIONS)
    for replication in range(BOOTSTRAP_REPLICATIONS):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate(
            [differential[start : start + block_length] for start in sampled_starts]
        )[:n]
        means[replication] = sample.mean()
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def comparison_row(
    window_size: int,
    horizon: int,
    actual: np.ndarray,
    model_forecast: np.ndarray,
    har_forecast: np.ndarray,
) -> dict[str, float | int | str]:
    model_mae = mean_absolute_error(actual, model_forecast)
    har_mae = mean_absolute_error(actual, har_forecast)
    differential = np.abs(actual - model_forecast) - np.abs(actual - har_forecast)
    ci_lower, ci_upper = bootstrap_confidence_interval(differential)
    return {
        "window_size": window_size,
        "horizon": horizon,
        "benchmark": "HAR-OLS",
                "n_obs": len(actual),
        "model_MAE": model_mae,
        "har_MAE": har_mae,
        "delta_MAE": model_mae - har_mae,
        "dm_p_value_unadjusted": dm_p_value(differential),
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
    }


# Model
from chronos import BaseChronosPipeline


pipeline = BaseChronosPipeline.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    device_map="cuda",
    torch_dtype=MODEL_DTYPE,
)
pipeline.model.eval()
print(f"Loaded {MODEL_ID} at {MODEL_REVISION}")


def predict_log_quantiles(contexts: np.ndarray) -> np.ndarray:
    batches = []
    start = 0
    batch_size = min(INFERENCE_BATCH_SIZE, len(contexts))
    while start < len(contexts):
        stop = min(start + batch_size, len(contexts))
        try:
            with torch.inference_mode():
                quantile_tensor, _ = pipeline.predict_quantiles(
                    inputs=torch.from_numpy(contexts[start:stop]),
                    prediction_length=MAX_HORIZON,
                    quantile_levels=QUANTILE_LEVELS,
                )
            if not isinstance(quantile_tensor, torch.Tensor):
                raise TypeError("The documented API must return one quantile tensor.")
            batch = quantile_tensor.detach().float().cpu().numpy()
            expected_shape = (stop - start, MAX_HORIZON, len(QUANTILE_LEVELS))
            if batch.shape != expected_shape:
                raise ValueError(f"Unexpected documented API shape {batch.shape}; expected {expected_shape}.")
            batches.append(batch)
            start = stop
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            warnings.warn(f"CUDA OOM: inference batch reduced to {batch_size}.", RuntimeWarning)
    return np.concatenate(batches)


# Window loop, validation and test
test_actual = target_matrix(vix_level, test_origins)
forecast_store: dict[int, np.ndarray] = {}
result_rows = []

for window_size in WINDOW_SIZES:
    print(f"Forecasting W={window_size}")
    contexts = build_contexts(log_vix, test_origins, window_size)
    log_quantiles = check_quantiles(predict_log_quantiles(contexts), len(test_origins))
    level_quantiles = safe_exp(log_quantiles[:, HORIZON_STEPS, :])
    forecast_store[window_size] = level_quantiles

    for horizon_index, horizon in enumerate(HORIZONS):
        positions = np.arange(0, len(test_origins), horizon)
        result_rows.append(
            comparison_row(
                window_size,
                horizon,
                test_actual[positions, horizon_index],
                level_quantiles[positions, horizon_index, 1],
                test_har[horizon][positions],
            )
        )


# Metrics and summary rows
# Only primary non-overlapping test comparisons feed the public output.


# CSV
OUTPUT_COLUMNS = [
    "window_size",
    "horizon",
    "benchmark",
    "n_obs",
    "model_MAE",
    "har_MAE",
    "delta_MAE",
    "dm_p_value_unadjusted",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
]
results = (
    pd.DataFrame(result_rows, columns=OUTPUT_COLUMNS)
    .sort_values(["window_size", "horizon"])
    .reset_index(drop=True)
)

RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
output_path = RESULTS_DIRECTORY / OUTPUT_FILENAME
results.to_csv(output_path, index=False)
print(f"Saved compact results CSV: {output_path}")


# Display
print("Primary test comparison, identical to the saved CSV")
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 180):
    display(results)

primary_har_ols = results


# Plots
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for horizon_index, horizon in enumerate(HORIZONS):
    horizon_rows = primary_har_ols[primary_har_ols["horizon"] == horizon].sort_values("window_size")
    axes[horizon_index].plot(horizon_rows["window_size"], horizon_rows["model_MAE"], marker="o", label=MODEL_FAMILY)
    axes[horizon_index].plot(horizon_rows["window_size"], horizon_rows["har_MAE"], marker="s", label="HAR-OLS")
    axes[horizon_index].set_title(f"MAE, h={horizon}")
    axes[horizon_index].set_xlabel("Window")
    axes[horizon_index].grid(alpha=0.3)
axes[0].legend()
fig.tight_layout()
plt.show()
plt.close(fig)

delta_matrix = primary_har_ols.pivot(index="window_size", columns="horizon", values="delta_MAE").reindex(
    index=WINDOW_SIZES,
    columns=HORIZONS,
)
fig, ax = plt.subplots(figsize=(6, 5))
image = ax.imshow(delta_matrix.to_numpy(), cmap="RdBu_r", aspect="auto")
ax.set_xticks(range(len(HORIZONS)), HORIZONS)
ax.set_yticks(range(len(WINDOW_SIZES)), WINDOW_SIZES)
ax.set_xlabel("Horizon")
ax.set_ylabel("Window")
ax.set_title("MAE difference vs HAR-OLS")
fig.colorbar(image, ax=ax, label="Model MAE - HAR MAE")
fig.tight_layout()
plt.show()
plt.close(fig)

for horizon_index, horizon in enumerate(HORIZONS):
    fig, axes = plt.subplots(len(WINDOW_SIZES), 1, figsize=(14, 13), sharex=True)
    plot_positions = np.arange(0, len(test_origins), horizon)
    target_dates = pd.to_datetime(dates[test_origins[plot_positions] + horizon])
    observed = test_actual[plot_positions, horizon_index]
    for axis, window_size in zip(axes, WINDOW_SIZES):
        quantiles = forecast_store[window_size][plot_positions, horizon_index]
        axis.plot(target_dates, observed, color="black", linewidth=1, label="Observed")
        axis.plot(target_dates, quantiles[:, 1], color="tab:blue", linewidth=1, label="q50")
        axis.plot(target_dates, test_har[horizon][plot_positions], color="tab:orange", linewidth=1, label="HAR-OLS")
        axis.fill_between(target_dates, quantiles[:, 0], quantiles[:, 2], color="tab:blue", alpha=0.18, label="q10-q90")
        axis.set_title(f"W={window_size}")
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=4)
    fig.suptitle(f"Non-overlapping test paths, horizon {horizon}")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


# Final assertions
assert results.shape == (len(WINDOW_SIZES) * len(HORIZONS), len(OUTPUT_COLUMNS))
assert list(results.columns) == OUTPUT_COLUMNS
assert len(forecast_store) == len(WINDOW_SIZES)
assert all(quantiles.shape == (len(test_origins), len(HORIZONS), 3) for quantiles in forecast_store.values())
assert all(np.isfinite(quantiles).all() for quantiles in forecast_store.values())
assert OUTPUT_FILENAME.count(".csv") == 1 and "_w" not in OUTPUT_FILENAME
print("All protocol assertions passed.")

"""### Fine Tuned"""

# Commented out IPython magic to ensure Python compatibility.

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q --upgrade-strategy only-if-needed "chronos-forecasting==2.3.1"

