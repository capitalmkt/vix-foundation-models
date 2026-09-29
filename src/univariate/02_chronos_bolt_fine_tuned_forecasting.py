"""
02. Chronos-Bolt - Fine-Tuned Forecasting (univariate)

Model family: Chronos-Bolt
Adaptation: Fine-Tuned Forecasting
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
Update configuration: Five fixed seeds, one fixed LoRA configuration, validation-selected optimizer updates and causal 80% refit.
LoRA configuration: rank 8, alpha 16, dropout 0.05 on the verified T5 attention q and v projections.
LoRA integration: local linear adapters operate on numeric attention projections without a token-embedding interface.
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
import gc
import hashlib
import importlib.metadata
import math
import platform
import random
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from chronos import BaseChronosPipeline
from IPython.display import display
from scipy.stats import t as student_t
from torch.utils.data import DataLoader, TensorDataset

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
SEEDS = [42, 123, 2024, 31415, 271828]
BOOTSTRAP_REPLICATIONS = 5000
BOOTSTRAP_SEED = 2026
OUTPUT_FILENAME = "forecast_chronos_bolt_fine_tuned_univariate.csv"
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


set_seed(SEEDS[0])


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
    print(f"VIX source: {path}")
    print(f"VIX SHA-256: {file_hash}")
    print(f"Dropped rows: {dropped_rows}")
    print(f"Range: {clean['Date'].iloc[0].date()} to {clean['Date'].iloc[-1].date()} | n={len(clean)}")
    return clean


vix_path = locate_approved_csv(LOCAL_VIX_PATHS, "VIX")
vix_data = load_vix_csv(vix_path)
dates = vix_data["Date"].to_numpy()
vix_level = vix_data["VIX"].to_numpy(dtype=np.float64)
log_vix = np.log(vix_level)
n_obs = len(vix_data)


# Split
validation_start = int(np.floor(0.60 * n_obs))
test_start = int(np.floor(0.80 * n_obs))
minimum_origin = max(MAX_WINDOW - 1, HAR_MONTH - 1)


def origin_range(start: int, stop: int) -> np.ndarray:
    return np.arange(max(start, minimum_origin), stop - MAX_HORIZON, dtype=int)


train_origins = origin_range(0, validation_start)
validation_origins = origin_range(validation_start, test_start)
refit_origins = origin_range(0, test_start)
test_origins = origin_range(test_start, n_obs)
if min(len(train_origins), len(refit_origins)) == 0:
    raise ValueError("The cleaned VIX history is too short for training and refit.")
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
    array = np.asarray(values, dtype=np.float64)
    clipped = np.clip(array, -20.0, 20.0)
    clip_count = int(np.count_nonzero(clipped != array))
    if clip_count:
        warnings.warn(f"Clipped {clip_count} log forecasts to [-20, 20] before exponentiation.", RuntimeWarning)
    return np.exp(clipped)


def check_quantiles(log_quantiles: np.ndarray, expected_rows: int) -> np.ndarray:
    quantiles = np.asarray(log_quantiles, dtype=np.float64)
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
            series.rolling(HAR_WEEK, min_periods=HAR_WEEK).mean().to_numpy(),
            series.rolling(HAR_MONTH, min_periods=HAR_MONTH).mean().to_numpy(),
        ]
    )


def expanding_har_ols(values: np.ndarray, origins: np.ndarray) -> dict[int, np.ndarray]:
    features = har_features(values)
    results: dict[int, np.ndarray] = {}
    first_training_origin = HAR_MONTH - 1
    for horizon in HORIZONS:
        ols_log = np.empty(len(origins), dtype=np.float64)
        for position, origin in enumerate(origins):
            training = np.arange(first_training_origin, origin - horizon + 1, dtype=int)
            assert len(training) and training[-1] + horizon <= origin
            x_train = features[training]
            y_train = values[training + horizon]
            finite = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)
            x_train, y_train = x_train[finite], y_train[finite]
            if len(y_train) < 60:
                raise ValueError(f"Only {len(y_train)} valid HAR rows are available at origin {origin}.")
            ols_fit = sm.OLS(y_train, x_train).fit()
            if not np.isfinite(ols_fit.params).all():
                raise FloatingPointError("HAR-OLS returned non-finite coefficients.")
            ols_log[position] = float(features[origin] @ ols_fit.params)
        results[horizon] = safe_exp(ols_log)
    return results


test_har = expanding_har_ols(log_vix, test_origins)


# Statistical comparison


def diebold_mariano_p_value(loss_differential: np.ndarray, effective_horizon: int = 1) -> float:
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
    raw_statistic = differential.mean() / math.sqrt(long_run_variance / n)
    hln_term = (n + 1.0 - 2.0 * effective_horizon + effective_horizon * (effective_horizon - 1.0) / n) / n
    statistic = float(raw_statistic * math.sqrt(max(hln_term, 0.0)))
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
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_REPLICATIONS, dtype=np.float64)
    blocks_needed = math.ceil(n / block_length)
    for replication in range(BOOTSTRAP_REPLICATIONS):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([differential[start : start + block_length] for start in sampled_starts])[:n]
        means[replication] = sample.mean()
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def primary_comparison_row(
    window_size: int,
    horizon: int,
    actual: np.ndarray,
    model_forecast: np.ndarray,
    har_forecast: np.ndarray,
) -> dict[str, Any]:
    model_mae = mean_absolute_error(actual, model_forecast)
    har_mae = mean_absolute_error(actual, har_forecast)
    differential = np.abs(actual - model_forecast) - np.abs(actual - har_forecast)
    lower, upper = moving_block_bootstrap_ci(differential)
    return {
        "window_size": window_size,
        "horizon": horizon,
        "benchmark": "HAR-OLS",
                "n_obs": len(actual),
        "model_MAE": model_mae,
        "har_MAE": har_mae,
        "delta_MAE": model_mae - har_mae,
        "dm_p_value_unadjusted": diebold_mariano_p_value(differential),
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
    }


# Model
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q", "v"]
LEARNING_RATE = 0.0001
TRAIN_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2
MIN_UPDATES = 100
VALIDATION_EVERY_UPDATES = 50
MAX_UPDATES = 500
EARLY_STOPPING_PATIENCE = 3
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0


class LoRALinear(torch.nn.Module):
    """Frozen linear projection plus a trainable low-rank residual."""

    def __init__(self, base_layer: torch.nn.Linear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.lora_dropout = torch.nn.Dropout(LORA_DROPOUT)
        self.lora_a = torch.nn.Linear(base_layer.in_features, LORA_R, bias=False)
        self.lora_b = torch.nn.Linear(LORA_R, base_layer.out_features, bias=False)
        self.scale = LORA_ALPHA / LORA_R
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b.weight)
        self.base_layer.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.lora_b(self.lora_a(self.lora_dropout(inputs)))
        return self.base_layer(inputs) + self.scale * residual


def add_lora_adapters(model: torch.nn.Module) -> list[str]:
    """Replace only Bolt's real q and v attention projections."""
    model.requires_grad_(False)
    replacements = []
    for full_name, module in list(model.named_modules()):
        leaf_name = full_name.rsplit(".", 1)[-1]
        if leaf_name not in LORA_TARGETS or not isinstance(module, torch.nn.Linear):
            continue
        parent_name, _, child_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module))
        replacements.append(full_name)
    missing = [target for target in LORA_TARGETS if not any(name.rsplit(".", 1)[-1] == target for name in replacements)]
    if missing:
        raise RuntimeError(f"LoRA targets not found in Chronos-Bolt: {missing}")
    return replacements


def verify_lora(model: torch.nn.Module) -> None:
    lora_parameters = [name for name, parameter in model.named_parameters() if "lora_" in name and parameter.requires_grad]
    trainable_parameters = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not lora_parameters:
        raise RuntimeError("No LoRA parameter is trainable. Check the fixed target modules.")
    unexpected = [name for name in trainable_parameters if "lora_" not in name]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected[:10]}")
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"LoRA trainable parameters: {trainable_count:,}/{total_count:,}")


def build_trainable_pipeline() -> BaseChronosPipeline:
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        device_map="cuda",
        torch_dtype=torch.float32,
    )
    replacements = add_lora_adapters(pipeline.model)
    pipeline.model.to(DEVICE)
    verify_lora(pipeline.model)
    print(f"LoRA projections replaced: {len(replacements)}")
    return pipeline


def predict_quantiles(pipeline: BaseChronosPipeline, contexts: np.ndarray) -> np.ndarray:
    pipeline.model.eval()
    batches = []
    for start in range(0, len(contexts), INFERENCE_BATCH_SIZE):
        context = torch.from_numpy(contexts[start : start + INFERENCE_BATCH_SIZE])
        with torch.inference_mode():
            quantiles, _ = pipeline.predict_quantiles(
                inputs=context,
                prediction_length=MAX_HORIZON,
                quantile_levels=QUANTILE_LEVELS,
            )
        if not isinstance(quantiles, torch.Tensor):
            raise TypeError("Expected a tensor from the documented quantile API.")
        batches.append(quantiles.detach().float().cpu().numpy())
    return np.concatenate(batches)


def make_loader(contexts: np.ndarray, targets: np.ndarray, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(contexts), torch.from_numpy(targets))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)


def supervised_arrays(origins: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    contexts = build_contexts(log_vix, origins, window_size)
    offsets = np.arange(1, MAX_HORIZON + 1, dtype=int)
    targets = log_vix[origins[:, None] + offsets].astype(np.float32)
    assert origins[-1] + MAX_HORIZON < n_obs
    return contexts, targets


def adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_adapter(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    trainable_names = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if set(state) != trainable_names:
        missing = sorted(trainable_names - set(state))
        unexpected = sorted(set(state) - trainable_names)
        raise RuntimeError(f"Invalid LoRA state. Missing={missing[:5]}, unexpected={unexpected[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))


def optimizer_updates(
    pipeline: BaseChronosPipeline,
    loader: DataLoader,
    maximum_updates: int,
    validation_callback: Any | None = None,
    snapshot_updates: set[int] | None = None,
) -> tuple[dict[int, dict[str, torch.Tensor]], int]:
    model = pipeline.model
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    iterator = iter(loader)
    completed = 0
    accumulated = 0
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    optimizer.zero_grad(set_to_none=True)
    while completed < maximum_updates:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        model.train()
        context, target = (tensor.to(DEVICE) for tensor in batch)
        loss = model(context=context, target=target).loss
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("Fine-tuning produced a non-finite loss.")
        (loss / GRADIENT_ACCUMULATION_STEPS).backward()
        accumulated += 1
        if accumulated < GRADIENT_ACCUMULATION_STEPS:
            continue
        torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed += 1
        accumulated = 0
        if snapshot_updates and completed in snapshot_updates:
            snapshots[completed] = adapter_state(model)
        if validation_callback and completed % VALIDATION_EVERY_UPDATES == 0 and validation_callback(completed):
            break
    del optimizer
    return snapshots, completed


def select_updates(window_size: int, seed: int) -> dict[int, int]:
    set_seed(seed)
    pipeline = build_trainable_pipeline()
    contexts, targets = supervised_arrays(train_origins, window_size)
    assert train_origins[-1] + MAX_HORIZON < validation_start
    loader = make_loader(contexts, targets, seed)
    validation_contexts = build_contexts(log_vix, validation_origins, window_size)
    best_mae = {horizon: np.inf for horizon in HORIZONS}
    best_update = {horizon: -1 for horizon in HORIZONS}
    checks_without_improvement = 0

    def validate(update: int) -> bool:
        nonlocal checks_without_improvement
        log_quantiles = check_quantiles(predict_quantiles(pipeline, validation_contexts), len(validation_origins))
        selected = log_quantiles[:, HORIZON_STEPS, :]
        level_quantiles = safe_exp(selected)
        improved = False
        for index, horizon in enumerate(HORIZONS):
            mae = mean_absolute_error(validation_actual[:, index], level_quantiles[:, index, 1])
            print(f"W={window_size} seed={seed} update={update} h={horizon} validation MAE={mae:.6f}")
            if update >= MIN_UPDATES and mae < best_mae[horizon] - 1e-10:
                best_mae[horizon] = mae
                best_update[horizon] = update
                improved = True
        checks_without_improvement = 0 if improved else checks_without_improvement + 1
        return update >= MIN_UPDATES and checks_without_improvement >= EARLY_STOPPING_PATIENCE

    optimizer_updates(pipeline, loader, MAX_UPDATES, validation_callback=validate)
    if any(update < 0 for update in best_update.values()):
        raise RuntimeError(f"No eligible update selected for W={window_size}, seed={seed}.")
    del pipeline, contexts, targets, loader, validation_contexts
    gc.collect()
    torch.cuda.empty_cache()
    return best_update


def check_selected_quantiles(values: np.ndarray) -> np.ndarray:
    expected = (values.shape[0], len(HORIZONS), len(QUANTILE_LEVELS))
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"Invalid selected quantile tensor: {values.shape}")
    if np.any(np.diff(values, axis=-1) < 0):
        values = np.sort(values, axis=-1)
    return values


def refit_and_forecast(window_size: int, seed: int, selected_updates: dict[int, int]) -> np.ndarray:
    set_seed(seed)
    pipeline = build_trainable_pipeline()
    contexts, targets = supervised_arrays(refit_origins, window_size)
    assert refit_origins[-1] + MAX_HORIZON < test_start
    loader = make_loader(contexts, targets, seed)
    required_updates = set(selected_updates.values())
    snapshots, completed = optimizer_updates(pipeline, loader, max(required_updates), snapshot_updates=required_updates)
    if completed != max(required_updates) or set(snapshots) != required_updates:
        raise RuntimeError("Refit did not preserve every required optimizer-update state.")
    test_contexts = build_contexts(log_vix, test_origins, window_size)
    horizon_quantiles = []
    for index, horizon in enumerate(HORIZONS):
        restore_adapter(pipeline.model, snapshots[selected_updates[horizon]])
        log_quantiles = check_quantiles(predict_quantiles(pipeline, test_contexts), len(test_origins))
        horizon_quantiles.append(log_quantiles[:, HORIZON_STEPS[index], :])
    selected = check_selected_quantiles(np.stack(horizon_quantiles, axis=1))
    del pipeline, contexts, targets, loader, test_contexts, snapshots
    gc.collect()
    torch.cuda.empty_cache()
    return selected


# Window loop, validation selection, causal refit and test
validation_actual = target_matrix(vix_level, validation_origins)
test_actual = target_matrix(vix_level, test_origins)
forecast_store: dict[int, dict[str, Any]] = {}
seed_diagnostics = []

for window_size in WINDOW_SIZES:
    print(f"Fine-tuning W={window_size}")
    test_by_seed = []
    for seed in SEEDS:
        selected_updates = select_updates(window_size, seed)
        test_by_seed.append(refit_and_forecast(window_size, seed, selected_updates))

    test_seed_matrix = np.stack(test_by_seed)
    test_ensemble_log = check_selected_quantiles(np.median(test_seed_matrix, axis=0))
    test_level = safe_exp(test_ensemble_log)
    test_seed_levels = safe_exp(test_seed_matrix[..., 1])

    for horizon_index, horizon in enumerate(HORIZONS):
        for seed_index, seed in enumerate(SEEDS):
            seed_diagnostics.append(
                {
                    "window_size": window_size,
                    "horizon": horizon,
                    "seed": seed,
                    "seed_MAE": mean_absolute_error(
                        test_actual[:, horizon_index],
                        test_seed_levels[seed_index, :, horizon_index],
                    ),
                }
            )

    forecast_store[window_size] = {"test_quantiles": test_level}
    del test_by_seed, test_seed_matrix, test_seed_levels
    gc.collect()
    torch.cuda.empty_cache()


# Metrics and summary rows
primary_rows = []
for window_size, stored in forecast_store.items():
    for horizon_index, horizon in enumerate(HORIZONS):
        positions = np.arange(0, len(test_origins), horizon, dtype=int)
        q50 = stored["test_quantiles"][positions, horizon_index, 1]
        primary_rows.append(
            primary_comparison_row(
                window_size,
                horizon,
                test_actual[positions, horizon_index],
                q50,
                test_har[horizon][positions],
            )
        )


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
    pd.DataFrame(primary_rows, columns=OUTPUT_COLUMNS)
    .sort_values(["window_size", "horizon"])
    .reset_index(drop=True)
)
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

delta_matrix = primary_har_ols.pivot(index="window_size", columns="horizon", values="delta_MAE").reindex(index=WINDOW_SIZES, columns=HORIZONS)
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
        quantiles = forecast_store[window_size]["test_quantiles"][plot_positions, horizon_index]
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

seed_plot_table = pd.DataFrame(seed_diagnostics)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for axis, horizon in zip(axes, HORIZONS):
    current = seed_plot_table[seed_plot_table["horizon"] == horizon]
    for seed in SEEDS:
        seed_rows = current[current["seed"] == seed].sort_values("window_size")
        axis.plot(seed_rows["window_size"], seed_rows["seed_MAE"], marker="o", alpha=0.65, label=str(seed))
    axis.set_title(f"Seed MAE, h={horizon}")
    axis.set_xlabel("Window")
    axis.grid(alpha=0.25)
axes[0].legend(title="Seed", ncol=2)
fig.tight_layout()
plt.show()
plt.close(fig)


# Final assertions
assert len(forecast_store) == len(WINDOW_SIZES)
assert all(forecast_store[window]["test_quantiles"].shape == (len(test_origins), len(HORIZONS), 3) for window in WINDOW_SIZES)
assert all(np.isfinite(forecast_store[window]["test_quantiles"]).all() for window in WINDOW_SIZES)
assert OUTPUT_FILENAME.count(".csv") == 1 and "_w" not in OUTPUT_FILENAME
print("All protocol assertions passed.")

"""### Classifier"""

# Commented out IPython magic to ensure Python compatibility.

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q --upgrade-strategy only-if-needed "chronos-forecasting==2.3.1"

