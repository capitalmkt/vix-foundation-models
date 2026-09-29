"""
04. Chronos-2 - Zero-Shot Forecasting (multivariate)

Model family: chronos_2
Adaptation: zero_shot
Information set: VIX+VVIX
Covariate method: native_past_covariate
Colab layout: run this complete file as one cell on a T4 GPU

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
from common.evaluation import arrange_quantiles
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
import torch
from IPython.display import display
from scipy.stats import t as student_t
from chronos import Chronos2Pipeline

# Configuration
MODEL_ID = "amazon/chronos-2"
MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
COVARIATE_METHOD = "native_past_covariate"
INFORMATION_SET = "VIX+VVIX"
WINDOW_SIZES = [16, 32, 64, 128, 256]
HORIZONS = [1, 5, 22]
HORIZON_STEPS = [0, 4, 21]
MAX_WINDOW = max(WINDOW_SIZES)
MAX_HORIZON = max(HORIZONS)
HAR_WEEK = 5
HAR_MONTH = 22
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
SEED = 2026
BOOTSTRAP_REPLICATIONS = 5000
BOOTSTRAP_SEED = SEED
OUTPUT_FILENAME = "forecast_chronos_2_zero_shot_multivariate.csv"
LOCAL_VIX_PATHS = [REPO_DATA_DIR / "VIX.csv", Path("/content/VIX.csv"), Path("/content/VIXCLS.csv"), Path("/content/vix.csv")]
LOCAL_VVIX_PATHS = [REPO_DATA_DIR / "VVIX.csv", Path("/content/VVIX.csv"), Path("/content/VVIXCLS.csv"), Path("/content/vvix.csv")]
RESULTS_DIRECTORY = resolve_results_directory()
enable_figure_autosave(RESULTS_DIRECTORY, Path(OUTPUT_FILENAME).stem)
if not torch.cuda.is_available():
    raise RuntimeError("Select a T4 GPU runtime in Google Colab before running this cell.")

# Reproducibility


set_seed(SEED)


# Data loading and alignment


def load_market_csv(path: Path, name: str) -> pd.DataFrame:
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = pd.read_csv(path)
    schemas = {
        "VIX": [("Date", "Close"), ("DATE", "VIXCLS")],
        "VVIX": [("Date", "Close"), ("DATE", "VVIX"), ("DATE", "VVIXCLS")],
    }
    selected = next(((date, value) for date, value in schemas[name] if {date, value}.issubset(raw.columns)), None)
    if selected is None:
        expected = " or ".join(f"{date},{value}" for date, value in schemas[name])
        raise ValueError(f"{name} CSV must use one of these schemas: {expected}.")
    date_column, value_column = selected
    parsed_dates = pd.to_datetime(raw[date_column], errors="coerce")
    try:
        parsed_dates = parsed_dates.dt.tz_localize(None)
    except TypeError:
        parsed_dates = pd.to_datetime(raw[date_column], errors="coerce", utc=True).dt.tz_localize(None)
    clean = pd.DataFrame(
        {
            "Date": parsed_dates.dt.normalize(),
            name: pd.to_numeric(raw[value_column], errors="coerce"),
        }
    )
    finite = np.isfinite(clean[name].to_numpy(dtype=np.float64, na_value=np.nan))
    valid = clean["Date"].notna() & clean[name].notna() & clean[name].gt(0) & finite
    dropped = int((~valid).sum())
    clean = clean.loc[valid].sort_values("Date").reset_index(drop=True)
    duplicates = clean.loc[clean["Date"].duplicated(keep=False), "Date"]
    if not duplicates.empty:
        raise ValueError(f"Duplicate {name} dates found: {duplicates.head().tolist()}")
    if clean.empty or not clean["Date"].is_monotonic_increasing:
        raise ValueError(f"Clean {name} observations must be nonempty and strictly chronological.")
    print(f"{name} source: {path}")
    print(f"{name} SHA-256: {file_hash} | dropped rows: {dropped} | cleaned rows: {len(clean)}")
    return clean


vix_path = locate_approved_csv(LOCAL_VIX_PATHS, "VIX")
vvix_path = locate_approved_csv(LOCAL_VVIX_PATHS, "VVIX")
vix_data = load_market_csv(vix_path, "VIX")
vvix_data = load_market_csv(vvix_path, "VVIX")
aligned_data = vix_data.merge(vvix_data, on="Date", how="inner", validate="one_to_one").sort_values("Date").reset_index(drop=True)
if aligned_data.empty or aligned_data["Date"].duplicated().any() or not aligned_data["Date"].is_monotonic_increasing:
    raise ValueError("The aligned VIX-VVIX sample must be nonempty, unique and chronological.")
aligned_hash = hashlib.sha256(pd.util.hash_pandas_object(aligned_data, index=False).values.tobytes()).hexdigest()
dates = aligned_data["Date"].to_numpy()
vix_level = aligned_data["VIX"].to_numpy(dtype=np.float64)
vvix_level = aligned_data["VVIX"].to_numpy(dtype=np.float64)
log_vix = np.log(vix_level)
log_vvix = np.log(vvix_level)
n_obs = len(aligned_data)
if not np.isfinite(log_vix).all() or not np.isfinite(log_vvix).all():
    raise FloatingPointError("Aligned log-VIX and log-VVIX must be finite.")
print(f"Aligned SHA-256: {aligned_hash}")
print(f"Aligned range: {aligned_data['Date'].iloc[0].date()} to {aligned_data['Date'].iloc[-1].date()} | n={n_obs}")
print(f"Information set: {INFORMATION_SET} | covariate method: {COVARIATE_METHOD}")


# Temporal split
test_start = int(np.floor(0.80 * n_obs))
minimum_origin = max(MAX_WINDOW - 1, HAR_MONTH - 1)
test_origins = np.arange(max(test_start, minimum_origin), n_obs - MAX_HORIZON, dtype=int)
if not len(test_origins):
    raise ValueError("The aligned history is too short for the fixed 60/20/20 protocol.")
assert test_origins[-1] + MAX_HORIZON < n_obs
print(f"Test origins: {len(test_origins)}")


# Contexts
def build_aligned_contexts(
    vix_values: np.ndarray,
    vvix_values: np.ndarray,
    origins: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = origins[:, None] + np.arange(-window_size + 1, 1, dtype=int)[None, :]
    if indices.min() < 0:
        raise ValueError("A context starts before the aligned sample.")
    np.testing.assert_array_equal(indices[:, -1], origins)
    vix_contexts = vix_values[indices].astype(np.float32, copy=False)
    vvix_contexts = vvix_values[indices].astype(np.float32, copy=False)
    if vix_contexts.shape != vvix_contexts.shape or vix_contexts.shape[1] != window_size:
        raise ValueError("VIX and VVIX contexts are misaligned.")
    if not np.isfinite(vix_contexts).all() or not np.isfinite(vvix_contexts).all():
        raise FloatingPointError("A model context contains non-finite values.")
    return vix_contexts, vvix_contexts


def target_matrix(values: np.ndarray, origins: np.ndarray) -> np.ndarray:
    return np.column_stack([values[origins + horizon] for horizon in HORIZONS])


# Numeric utility shared by forecasting and HAR benchmarks
def safe_exp(values: np.ndarray) -> tuple[np.ndarray, int]:
    array = np.asarray(values, dtype=np.float64)
    clipped = np.clip(array, -20.0, 20.0)
    clip_count = int(np.count_nonzero(clipped != array))
    if clip_count:
        warnings.warn(f"Clipped {clip_count} log forecasts before exponentiation.", RuntimeWarning)
    return np.exp(clipped), clip_count

# Forecast utilities


# Shared HAR-OLS and HAR-X-OLS forecasts
def har_design(target_values: np.ndarray, covariate_values: np.ndarray | None) -> np.ndarray:
    target_series = pd.Series(target_values)
    columns = [
        np.ones(len(target_values)),
        target_values,
        target_series.rolling(HAR_WEEK, min_periods=HAR_WEEK).mean().to_numpy(),
        target_series.rolling(HAR_MONTH, min_periods=HAR_MONTH).mean().to_numpy(),
    ]
    if covariate_values is not None:
        covariate_series = pd.Series(covariate_values)
        columns.extend(
            [
                covariate_values,
                covariate_series.rolling(HAR_WEEK, min_periods=HAR_WEEK).mean().to_numpy(),
                covariate_series.rolling(HAR_MONTH, min_periods=HAR_MONTH).mean().to_numpy(),
            ]
        )
    return np.column_stack(columns)


def expanding_direct_ols(design: np.ndarray, target_values: np.ndarray, origins: np.ndarray, label: str) -> np.ndarray:
    predictions = np.empty((len(origins), len(HORIZONS)), dtype=np.float64)
    fallback_count = 0
    for horizon_index, horizon in enumerate(HORIZONS):
        for position, origin in enumerate(origins):
            training = np.arange(HAR_MONTH - 1, origin - horizon + 1, dtype=int)
            x = design[training]
            y = target_values[training + horizon]
            finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
            x, y = x[finite], y[finite]
            if len(y) < 60:
                raise ValueError(f"{label} has only {len(y)} observable rows at origin {origin}, h={horizon}.")
            rank = np.linalg.matrix_rank(x)
            condition = np.linalg.cond(x)
            if rank == x.shape[1] and np.isfinite(condition) and condition <= 1e12:
                coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
            else:
                scale = max(float(np.trace(x.T @ x) / x.shape[1]), 1.0)
                penalty = np.eye(x.shape[1]) * (1e-8 * scale)
                penalty[0, 0] = 0.0
                coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ y)
                fallback_count += 1
            if not np.isfinite(coefficients).all():
                raise FloatingPointError(f"{label} returned non-finite coefficients.")
            predictions[position, horizon_index] = float(design[origin] @ coefficients)
    if fallback_count:
        warnings.warn(f"{label}: ridge fallback used for {fallback_count} ill-conditioned fits.", RuntimeWarning)
    return predictions


har_ols_log = expanding_direct_ols(har_design(log_vix, None), log_vix, test_origins, "HAR-OLS")
har_x_ols_log = expanding_direct_ols(har_design(log_vix, log_vvix), log_vix, test_origins, "HAR-X-OLS")
har_ols_level, _ = safe_exp(har_ols_log)
har_x_ols_level, _ = safe_exp(har_x_ols_log)
test_benchmarks = {
    "HAR-OLS": har_ols_level,
    "HAR-X-OLS": har_x_ols_level,
}
test_actual = target_matrix(vix_level, test_origins)


# Statistical tests
def dm_p_value(loss_differential: np.ndarray, hac_lags: int | None = None) -> float:
    differential = np.asarray(loss_differential, dtype=np.float64)
    if differential.ndim != 1 or not np.isfinite(differential).all():
        raise ValueError("DM requires a finite one-dimensional loss differential.")
    n = len(differential)
    if n < 3:
        return np.nan
    lag = min(max(0, int(np.floor(n ** (1 / 3))) if hac_lags is None else int(hac_lags)), n - 1)
    centered = differential - differential.mean()
    long_run_variance = float(centered @ centered / n)
    for current_lag in range(1, lag + 1):
        covariance = float(centered[current_lag:] @ centered[:-current_lag] / n)
        long_run_variance += 2.0 * (1.0 - current_lag / (lag + 1.0)) * covariance
    if long_run_variance <= np.finfo(float).eps:
        return np.nan
    statistic = float(differential.mean() / math.sqrt(long_run_variance / n))
    return float(2.0 * student_t.sf(abs(statistic), df=n - 1))


def moving_block_bootstrap(loss_differential: np.ndarray, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    differential = np.asarray(loss_differential, dtype=np.float64)
    if differential.ndim != 1 or not np.isfinite(differential).all():
        raise ValueError("Bootstrap requires finite paired losses.")
    n = len(differential)
    if n < 2:
        return float(differential.mean()), float(differential.mean())
    block_length = min(max(2, int(round(n ** (1 / 3)))), n)
    starts = np.arange(n - block_length + 1)
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_REPLICATIONS, dtype=np.float64)
    blocks_needed = math.ceil(n / block_length)
    for replication in range(BOOTSTRAP_REPLICATIONS):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([differential[start : start + block_length] for start in sampled_starts])[:n]
        means[replication] = sample.mean()
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


# Native Chronos-2 past-covariate inference
MODEL_DISPLAY_NAME = "Chronos-2+VVIX"
INTERVAL_LABEL = "native_past_covariate_quantiles"
INFERENCE_BATCH_SIZE = 64
print(
    f"Python {platform.python_version()} | PyTorch {torch.__version__} | "
    f"chronos-forecasting {importlib.metadata.version('chronos-forecasting')} | GPU {torch.cuda.get_device_name(0)}"
)
pipeline = Chronos2Pipeline.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, device_map="cuda", torch_dtype=torch.float32
)
pipeline.model.eval()
print(f"Loaded {MODEL_ID} at {MODEL_REVISION}")


def build_native_inputs(origins: np.ndarray, window_size: int) -> list[dict[str, object]]:
    vix_contexts, vvix_contexts = build_aligned_contexts(log_vix, log_vvix, origins, window_size)
    return [
        {"target": vix_contexts[index], "past_covariates": {"log_vvix": vvix_contexts[index]}}
        for index in range(len(origins))
    ]


def predict_native_quantiles(origins: np.ndarray, window_size: int) -> np.ndarray:
    inputs = build_native_inputs(origins, window_size)
    quantile_list, _ = pipeline.predict_quantiles(
        inputs=inputs,
        prediction_length=MAX_HORIZON,
        quantile_levels=QUANTILE_LEVELS,
        batch_size=INFERENCE_BATCH_SIZE,
        context_length=window_size,
        cross_learning=False,
    )
    if not isinstance(quantile_list, list) or len(quantile_list) != len(origins):
        raise TypeError("Chronos-2 must return one quantile tensor per independent origin.")
    output = torch.stack([tensor[0] for tensor in quantile_list]).detach().float().cpu().numpy()
    output, crossings = arrange_quantiles(output, (len(origins), MAX_HORIZON, 3))
    if crossings:
        print(f"W={window_size}: repaired {crossings} Chronos-2 quantile crossings")
    return output


forecast_store: dict[int, np.ndarray] = {}
for window_size in WINDOW_SIZES:
    print(f"Forecasting W={window_size}")
    test_log = predict_native_quantiles(test_origins, window_size)[:, HORIZON_STEPS]
    test_level, test_clips = safe_exp(test_log)
    forecast_store[window_size] = test_level
    print(f"W={window_size} clips={test_clips}")

# Metrics, compact results and robustness checks
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


def probabilistic_summary(actual: np.ndarray, quantiles: np.ndarray) -> tuple[float, float, float, float, float, float]:
    q10, q50, q90 = quantiles.T
    losses = []
    for level, forecast in zip(QUANTILE_LEVELS, [q10, q50, q90]):
        error = actual - forecast
        losses.append(float(np.mean(np.maximum(level * error, (level - 1.0) * error))))
    coverage = float(np.mean((actual >= q10) & (actual <= q90)))
    width = q90 - q10
    interval_score = width + 10.0 * (q10 - actual) * (actual < q10) + 10.0 * (actual - q90) * (actual > q90)
    return *losses, coverage, float(np.mean(width)), float(np.mean(interval_score))


rows: list[dict[str, object]] = []
for window_size in WINDOW_SIZES:
    level_quantiles = forecast_store[window_size]
    for horizon_index, horizon in enumerate(HORIZONS):
        primary_positions = np.arange(0, len(test_origins), horizon, dtype=int)
        actual = test_actual[primary_positions, horizon_index]
        model_forecast = level_quantiles[primary_positions, horizon_index, 1]
        for benchmark_name, benchmark_matrix in test_benchmarks.items():
            benchmark_forecast = benchmark_matrix[primary_positions, horizon_index]
            differential = np.abs(actual - model_forecast) - np.abs(actual - benchmark_forecast)
            lower, upper = moving_block_bootstrap(differential)
            model_mae = float(np.mean(np.abs(actual - model_forecast)))
            benchmark_mae = float(np.mean(np.abs(actual - benchmark_forecast)))
            rows.append(
                {
                    "window_size": window_size,
                    "horizon": horizon,
                    "benchmark": benchmark_name,
                    "n_obs": len(actual),
                    "model_MAE": model_mae,
                    "har_MAE": benchmark_mae,
                    "delta_MAE": model_mae - benchmark_mae,
                    "dm_p_value_unadjusted": dm_p_value(differential),
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                }
            )
        diagnostics = probabilistic_summary(test_actual[:, horizon_index], level_quantiles[:, horizon_index])
        print(
            f"W={window_size} h={horizon} {INTERVAL_LABEL} | "
            f"pinball(q10/q50/q90)={diagnostics[0]:.4f}/{diagnostics[1]:.4f}/{diagnostics[2]:.4f} | "
            f"coverage={diagnostics[3]:.3f} width={diagnostics[4]:.3f} interval_score={diagnostics[5]:.3f}"
        )


results = pd.DataFrame(rows, columns=OUTPUT_COLUMNS).sort_values(
    ["window_size", "horizon", "benchmark"], kind="stable"
).reset_index(drop=True)
assert list(results.columns) == OUTPUT_COLUMNS
assert len(results) == len(WINDOW_SIZES) * len(HORIZONS) * 2
assert set(results["window_size"].astype(int)) == set(WINDOW_SIZES)
assert set(results["horizon"].astype(int)) == set(HORIZONS)
assert set(results["benchmark"]) == {"HAR-OLS", "HAR-X-OLS"}
assert not results.duplicated(["window_size", "horizon", "benchmark"]).any()
np.testing.assert_allclose(results["delta_MAE"], results["model_MAE"] - results["har_MAE"])
assert np.isfinite(results[["model_MAE", "har_MAE", "delta_MAE", "bootstrap_ci_lower", "bootstrap_ci_upper"]]).all().all()
assert results["dm_p_value_unadjusted"].dropna().between(0.0, 1.0).all()


# Single CSV export and the only displayed DataFrame
RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
csv_path = RESULTS_DIRECTORY / OUTPUT_FILENAME
results.to_csv(csv_path, index=False)
print(f"Saved compact results CSV: {csv_path}")
display(results)


# Forecast plots: exactly five figures, or six for fine-tuning
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
for axis, horizon in zip(axes, HORIZONS):
    current = results[results["horizon"] == horizon]
    model_line = current.drop_duplicates("window_size").sort_values("window_size")
    axis.plot(model_line["window_size"], model_line["model_MAE"], marker="o", label=MODEL_DISPLAY_NAME)
    for benchmark_name, marker in [("HAR-OLS", "s"), ("HAR-X-OLS", "^")]:
        benchmark_line = current[current["benchmark"] == benchmark_name].sort_values("window_size")
        axis.plot(benchmark_line["window_size"], benchmark_line["har_MAE"], marker=marker, label=benchmark_name)
    axis.set_title(f"MAE, h={horizon}")
    axis.set_xlabel("Window")
    axis.grid(alpha=0.25)
axes[0].set_ylabel("MAE")
axes[0].legend()
fig.tight_layout()
plt.show()
plt.close(fig)

heatmaps = []
for benchmark_name in ["HAR-OLS", "HAR-X-OLS"]:
    heatmaps.append(
        results[results["benchmark"] == benchmark_name]
        .pivot(index="window_size", columns="horizon", values="delta_MAE")
        .reindex(index=WINDOW_SIZES, columns=HORIZONS)
        .to_numpy()
    )
limit = max(float(np.max(np.abs(matrix))) for matrix in heatmaps)
limit = max(limit, np.finfo(float).eps)
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for axis, benchmark_name, matrix in zip(axes, ["HAR-OLS", "HAR-X-OLS"], heatmaps):
    image = axis.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit)
    axis.set_xticks(range(len(HORIZONS)), HORIZONS)
    axis.set_yticks(range(len(WINDOW_SIZES)), WINDOW_SIZES)
    axis.set_xlabel("Horizon")
    axis.set_title(f"Model MAE - {benchmark_name} MAE")
axes[0].set_ylabel("Window")
fig.colorbar(image, ax=axes.ravel().tolist(), label="Delta MAE")
fig.subplots_adjust(wspace=0.2, right=0.88)
plt.show()
plt.close(fig)

for horizon_index, horizon in enumerate(HORIZONS):
    positions = np.arange(0, len(test_origins), horizon, dtype=int)
    target_dates = pd.to_datetime(dates[test_origins[positions] + horizon])
    observed = test_actual[positions, horizon_index]
    fig, axes = plt.subplots(len(WINDOW_SIZES), 1, figsize=(14, 13), sharex=True)
    for axis, window_size in zip(axes, WINDOW_SIZES):
        quantiles = forecast_store[window_size][positions, horizon_index]
        axis.plot(target_dates, observed, color="black", linewidth=1, label="Observed")
        axis.plot(target_dates, quantiles[:, 1], color="tab:blue", linewidth=1, label=MODEL_DISPLAY_NAME)
        axis.plot(target_dates, test_benchmarks["HAR-OLS"][positions, horizon_index], color="tab:orange", linewidth=1, label="HAR-OLS")
        axis.plot(target_dates, test_benchmarks["HAR-X-OLS"][positions, horizon_index], color="tab:green", linewidth=1, label="HAR-X-OLS")
        axis.fill_between(target_dates, quantiles[:, 0], quantiles[:, 2], color="tab:blue", alpha=0.16, label="q10-q90")
        axis.set_title(f"W={window_size}")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=5, fontsize=8)
    fig.suptitle(f"Non-overlapping test paths, h={horizon}")
    fig.tight_layout()
    plt.show()
    plt.close(fig)

# Final assertions
assert len(forecast_store) == len(WINDOW_SIZES)
assert all(forecast_store[window].shape == (len(test_origins), len(HORIZONS), 3) for window in WINDOW_SIZES)
assert all(np.isfinite(forecast_store[window]).all() for window in WINDOW_SIZES)
assert "row_type" not in results.columns and "covariate_method" not in results.columns and "MASE" not in results.columns
print("All multivariate forecasting assertions passed.")

"""### Fine Tuned"""

# Commented out IPython magic to ensure Python compatibility.

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q --upgrade-strategy only-if-needed "chronos-forecasting==2.3.1"

