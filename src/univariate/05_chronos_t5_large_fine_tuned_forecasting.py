"""
05. Chronos T5 Large - Fine-Tuned Forecasting (univariate)

Chronos T5 Large fine-tuning with univariate VIX input and a causal 80/20 split. Fixed windows: 32, 64, 128, 256. Fixed horizons: 1, 5, 22 trading days. Model checkpoint: amazon/chronos-t5-large@0e46c9c7e2e9f74b53db0617fdfcfe42a413e54a. The cell applies one LoRA setup for 50 updates, requires an approved CSV and runs on a Colab T4 GPU.

Documentation:
  https://github.com/amazon-science/chronos-forecasting
  https://huggingface.co/amazon/chronos-t5-large/commit/0e46c9c7e2e9f74b53db0617fdfcfe42a413e54a

Installation notes:
  pip install "chronos-forecasting==2.3.1" "peft==0.18.1"

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
import gc
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
MODEL_FAMILY = "chronos_t5_large"
MODEL_ID = "amazon/chronos-t5-large"
MODEL_REVISION = "0e46c9c7e2e9f74b53db0617fdfcfe42a413e54a"
WINDOW_SIZES = [32, 64, 128, 256]
HORIZONS = [1, 5, 22]
HORIZON_STEPS = [0, 4, 21]
MAX_HORIZON = 22
HAR_WEEK = 5
HAR_MONTH = 22
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
INFERENCE_BATCH_SIZE = 8
SEED = 123
BOOTSTRAP_REPLICATIONS = 5000
BOOTSTRAP_SEED = 2026
OUTPUT_FILENAME = "forecast_chronos_t5_large_fine_tuned_univariate.csv"
LOCAL_VIX_PATHS = [REPO_DATA_DIR / "VIX.csv", Path("/content/VIX.csv"), Path("/content/VIXCLS.csv"), Path("/content/vix.csv")]
RESULTS_DIRECTORY = resolve_results_directory()
enable_figure_autosave(RESULTS_DIRECTORY, Path(OUTPUT_FILENAME).stem)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = pd.read_csv(path)
    if {"Date", "Close"}.issubset(raw.columns):
        date_column, value_column = "Date", "Close"
    elif {"DATE", "VIXCLS"}.issubset(raw.columns):
        date_column, value_column = "DATE", "VIXCLS"
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
    if not clean["Date"].is_monotonic_increasing:
        raise AssertionError("VIX dates are not strictly chronological.")

    print(f"VIX source: {path}")
    print(f"VIX SHA-256: {file_hash}")
    print(f"Dropped rows: {dropped_rows}")
    print(f"Range: {clean['Date'].iloc[0].date()} to {clean['Date'].iloc[-1].date()} | n={len(clean)}")
    return clean


vix_data = load_vix_csv(locate_approved_csv(LOCAL_VIX_PATHS, "VIX"))
dates = vix_data["Date"].to_numpy()
vix_level = vix_data["VIX"].to_numpy(dtype=np.float64)
log_vix = np.log(vix_level)
n_obs = len(vix_data)


# Split
test_start = int(np.floor(0.80 * n_obs))
minimum_origin = max(max(WINDOW_SIZES) - 1, HAR_MONTH - 1)
test_origins = np.arange(max(test_start, minimum_origin), n_obs - MAX_HORIZON, dtype=int)
refit_origins = np.arange(minimum_origin, test_start - MAX_HORIZON, dtype=int)
if min(len(refit_origins), len(test_origins)) == 0:
    raise ValueError("The cleaned VIX history is too short for the fixed 80/20 protocol.")
assert refit_origins[-1] + MAX_HORIZON < test_start
assert test_origins[-1] + MAX_HORIZON < n_obs
print(f"Refit origins: {len(refit_origins)} | Test origins: {len(test_origins)}")


# Contexts
def build_contexts(values: np.ndarray, origins: np.ndarray, window_size: int) -> np.ndarray:
    indices = origins[:, None] + np.arange(-window_size + 1, 1)
    assert indices.min() >= 0
    np.testing.assert_array_equal(indices[:, -1], origins)
    contexts = values[indices].astype(np.float32, copy=False)
    assert np.isfinite(contexts).all()
    return contexts


def target_matrix(values: np.ndarray, origins: np.ndarray) -> np.ndarray:
    return np.column_stack([values[origins + horizon] for horizon in HORIZONS])


def safe_exp(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    clipped = np.clip(array, -20.0, 20.0)
    clip_count = int(np.count_nonzero(clipped != array))
    if clip_count:
        warnings.warn(f"Clipped {clip_count} log forecasts to [-20, 20] before exponentiation.", RuntimeWarning)
    return np.exp(clipped)


def validate_quantiles(values: np.ndarray, expected_rows: int) -> np.ndarray:
    quantiles = np.asarray(values, dtype=np.float64)
    expected_shape = (expected_rows, MAX_HORIZON, len(QUANTILE_LEVELS))
    if quantiles.shape != expected_shape:
        raise ValueError(f"Unexpected quantile shape {quantiles.shape}; expected {expected_shape}.")
    if not np.isfinite(quantiles).all():
        raise FloatingPointError("The model returned NaN or infinite quantiles.")
    return np.sort(quantiles, axis=-1) if np.any(np.diff(quantiles, axis=-1) < 0) else quantiles


# HAR benchmark
def har_features(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return np.column_stack(
        [
            np.ones(len(values)),
            values,
            series.rolling(HAR_WEEK, min_periods=HAR_WEEK).mean().to_numpy(),
            series.rolling(HAR_MONTH, min_periods=HAR_MONTH).mean().to_numpy(),
        ]
    )


def expanding_har_ols(values: np.ndarray, origins: np.ndarray) -> dict[int, np.ndarray]:
    features = har_features(values)
    forecasts = {}
    for horizon in HORIZONS:
        log_forecast = np.empty(len(origins), dtype=np.float64)
        for position, origin in enumerate(origins):
            training = np.arange(HAR_MONTH - 1, origin - horizon + 1, dtype=int)
            assert len(training) and training[-1] + horizon <= origin
            x_train = features[training]
            y_train = values[training + horizon]
            finite = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)
            x_train, y_train = x_train[finite], y_train[finite]
            if len(y_train) < 60:
                raise ValueError(f"Only {len(y_train)} valid HAR rows are available at origin {origin}.")
            fit = sm.OLS(y_train, x_train).fit()
            if not np.isfinite(fit.params).all():
                raise FloatingPointError("HAR-OLS returned non-finite coefficients.")
            log_forecast[position] = float(features[origin] @ fit.params)
        forecasts[horizon] = safe_exp(log_forecast)
    return forecasts


test_har = expanding_har_ols(log_vix, test_origins)


# Statistical comparison
def diebold_mariano_p_value(loss_differential: np.ndarray) -> float:
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


def moving_block_bootstrap_ci(loss_differential: np.ndarray) -> tuple[float, float]:
    differential = np.asarray(loss_differential, dtype=np.float64)
    if differential.ndim != 1 or not np.isfinite(differential).all():
        raise ValueError("Bootstrap requires a finite one-dimensional loss differential.")
    n = len(differential)
    if n < 2:
        return np.nan, np.nan

    block_length = min(max(2, int(round(n ** (1 / 3)))), n)
    starts = np.arange(n - block_length + 1)
    blocks_needed = math.ceil(n / block_length)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_REPLICATIONS, dtype=np.float64)
    for replication in range(BOOTSTRAP_REPLICATIONS):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([differential[start : start + block_length] for start in sampled_starts])[:n]
        means[replication] = sample.mean()
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def comparison_row(
    window_size: int,
    horizon: int,
    actual: np.ndarray,
    model_forecast: np.ndarray,
    har_forecast: np.ndarray,
) -> dict[str, object]:
    model_mae = float(np.mean(np.abs(actual - model_forecast)))
    har_mae = float(np.mean(np.abs(actual - har_forecast)))
    differential = np.abs(actual - model_forecast) - np.abs(actual - har_forecast)
    ci_lower, ci_upper = moving_block_bootstrap_ci(differential)
    return {
        "window_size": window_size,
        "horizon": horizon,
        "benchmark": "HAR-OLS",
        "n_obs": len(actual),
        "model_MAE": model_mae,
        "har_MAE": har_mae,
        "delta_MAE": model_mae - har_mae,
        "dm_p_value_unadjusted": diebold_mariano_p_value(differential),
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
    }


# Model
from chronos import ChronosPipeline
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, TensorDataset


LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q", "v"]
LEARNING_RATE = 0.0001
TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
FIXED_UPDATES = 50
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
PROGRESS_EVERY_UPDATES = 10


def verify_lora(model: torch.nn.Module) -> None:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not any("lora_" in name for name, _ in trainable):
        raise RuntimeError("No LoRA parameter is trainable. Check the fixed target modules.")
    unexpected = [name for name, _ in trainable if "lora_" not in name]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected[:10]}")
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"LoRA trainable parameters: {trainable_count:,}/{total_count:,}")


def build_trainable_pipeline() -> ChronosPipeline:
    pipeline = ChronosPipeline.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        device_map="cuda",
        torch_dtype=torch.float32,
    )
    base_model = pipeline.model.model
    available_targets = {name.rsplit(".", 1)[-1] for name, _ in base_model.named_modules()}
    missing_targets = set(LORA_TARGETS) - available_targets
    if missing_targets:
        raise RuntimeError(f"LoRA targets not found: {sorted(missing_targets)}")

    pipeline.model.model = get_peft_model(
        base_model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        ),
    ).to(DEVICE)
    verify_lora(pipeline.model.model)
    return pipeline


def training_loss(pipeline: ChronosPipeline, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    context, target = batch
    prediction_length = int(pipeline.model.config.prediction_length)
    native_target = torch.full((len(target), prediction_length), torch.nan, dtype=torch.float32)
    native_target[:, :MAX_HORIZON] = target
    input_ids, attention_mask, scale = pipeline.tokenizer.context_input_transform(context)
    labels, label_mask = pipeline.tokenizer.label_input_transform(native_target, scale)
    labels[label_mask == 0] = -100
    return pipeline.model.model(
        input_ids=input_ids.to(DEVICE),
        attention_mask=attention_mask.to(DEVICE),
        labels=labels.to(DEVICE),
    ).loss


def fine_tune(pipeline: ChronosPipeline, loader: DataLoader) -> None:
    model = pipeline.model.model
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    iterator = iter(loader)
    model.train()

    for update in range(1, FIXED_UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(GRADIENT_ACCUMULATION_STEPS):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            loss = training_loss(pipeline, batch)
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError("Fine-tuning produced a non-finite loss.")
            (loss / GRADIENT_ACCUMULATION_STEPS).backward()
        torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        optimizer.step()
        if update % PROGRESS_EVERY_UPDATES == 0 or update == FIXED_UPDATES:
            print(f"Optimizer updates: {update}/{FIXED_UPDATES}")

    del optimizer


def predict_quantiles(pipeline: ChronosPipeline, contexts: np.ndarray) -> np.ndarray:
    pipeline.model.model.eval()
    batches = []
    for start in range(0, len(contexts), INFERENCE_BATCH_SIZE):
        set_seed(SEED + start)
        context = torch.from_numpy(contexts[start : start + INFERENCE_BATCH_SIZE])
        with torch.inference_mode():
            quantiles, _ = pipeline.predict_quantiles(
                inputs=context,
                prediction_length=MAX_HORIZON,
                quantile_levels=QUANTILE_LEVELS,
                num_samples=20,
            )
        if not isinstance(quantiles, torch.Tensor):
            raise TypeError("Expected a tensor from the documented quantile API.")
        batches.append(quantiles.detach().float().cpu().numpy())
    return np.concatenate(batches)


def refit_and_forecast(window_size: int, seed: int) -> np.ndarray:
    set_seed(seed)
    pipeline = build_trainable_pipeline()
    contexts = build_contexts(log_vix, refit_origins, window_size)
    offsets = np.arange(1, MAX_HORIZON + 1)
    targets = log_vix[refit_origins[:, None] + offsets].astype(np.float32)
    assert refit_origins[-1] + MAX_HORIZON < test_start
    loader = DataLoader(
        TensorDataset(torch.from_numpy(contexts), torch.from_numpy(targets)),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    fine_tune(pipeline, loader)

    test_contexts = build_contexts(log_vix, test_origins, window_size)
    log_quantiles = validate_quantiles(predict_quantiles(pipeline, test_contexts), len(test_origins))
    selected = log_quantiles[:, HORIZON_STEPS, :]
    expected_shape = (len(test_origins), len(HORIZONS), len(QUANTILE_LEVELS))
    if selected.shape != expected_shape:
        raise ValueError(f"Invalid selected quantile tensor: {selected.shape}")

    del pipeline, contexts, targets, loader, test_contexts, log_quantiles
    gc.collect()
    torch.cuda.empty_cache()
    return selected


# Window loop, fixed-update causal refit and test
test_actual = target_matrix(vix_level, test_origins)
forecast_store = {}

for window_size in WINDOW_SIZES:
    print(f"Fine-tuning W={window_size} with {FIXED_UPDATES} fixed updates")
    test_log = refit_and_forecast(window_size, SEED)
    forecast_store[window_size] = safe_exp(test_log)
    del test_log
    gc.collect()
    torch.cuda.empty_cache()


# Metrics and compact CSV rows
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
result_rows = []
for window_size in WINDOW_SIZES:
    quantiles = forecast_store[window_size]
    for horizon_index, horizon in enumerate(HORIZONS):
        positions = np.arange(0, len(test_origins), horizon, dtype=int)
        result_rows.append(
            comparison_row(
                window_size,
                horizon,
                test_actual[positions, horizon_index],
                quantiles[positions, horizon_index, 1],
                test_har[horizon][positions],
            )
        )

results = pd.DataFrame(result_rows, columns=OUTPUT_COLUMNS).sort_values(["window_size", "horizon"]).reset_index(drop=True)
assert len(results) == len(WINDOW_SIZES) * len(HORIZONS)
assert list(results.columns) == OUTPUT_COLUMNS

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
    plot_positions = np.arange(0, len(test_origins), horizon, dtype=int)
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
assert len(forecast_store) == len(WINDOW_SIZES)
assert all(values.shape == (len(test_origins), len(HORIZONS), 3) for values in forecast_store.values())
assert all(np.isfinite(values).all() for values in forecast_store.values())
assert OUTPUT_FILENAME.count(".csv") == 1 and "_w" not in OUTPUT_FILENAME
print("All protocol assertions passed.")

"""### Classifier

"""

# Commented out IPython magic to ensure Python compatibility.

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q --upgrade-strategy only-if-needed "chronos-forecasting==2.3.1" "peft==0.18.1"

