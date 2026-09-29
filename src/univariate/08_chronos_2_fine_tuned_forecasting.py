"""
08. Chronos-2 - Fine-Tuned Forecasting (univariate)

Chronos-2 fine-tuned forecasting with univariate VIX input. Fixed windows: 16, 32, 64, 128, 256. Fixed horizons: 1, 5, 22 trading days. Three LoRA seeds use validation-selected updates followed by a causal 80% refit. Model checkpoint: amazon/chronos-2@29ec3766d36d6f73f0696f85560a422f50e8498c.

Documentation:
  https://github.com/amazon-science/chronos-forecasting
  https://huggingface.co/amazon/chronos-2/commit/29ec3766d36d6f73f0696f85560a422f50e8498c

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
import gc
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
from torch.utils.data import DataLoader, TensorDataset

# Configuration
MODEL_FAMILY = "chronos_2"
MODEL_ID = "amazon/chronos-2"
MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
WINDOW_SIZES = [16, 32, 64, 128, 256]
HORIZONS = [1, 5, 22]
HORIZON_STEPS = [0, 4, 21]
MAX_HORIZON = 22
MAX_WINDOW = 256
HAR_WEEK = 5
HAR_MONTH = 22
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
INFERENCE_BATCH_SIZE = 64
SEEDS = [42, 123, 2024]
BOOTSTRAP_REPLICATIONS = 5000
BOOTSTRAP_SEED = 2026
OUTPUT_FILENAME = "forecast_chronos_2_fine_tuned_univariate.csv"
LOCAL_VIX_PATHS = [REPO_DATA_DIR / "VIX.csv", Path("/content/VIX.csv"), Path("/content/VIXCLS.csv"), Path("/content/vix.csv")]
RESULTS_DIRECTORY = resolve_results_directory()
enable_figure_autosave(RESULTS_DIRECTORY, Path(OUTPUT_FILENAME).stem)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = {"q", "k", "v", "o"}
LEARNING_RATE = 5e-05
TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
MIN_UPDATES = 100
VALIDATION_EVERY_UPDATES = 50
MAX_UPDATES = 500
EARLY_STOPPING_PATIENCE = 3
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

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
    raw = pd.read_csv(path)
    if {"Date", "Close"}.issubset(raw.columns):
        date_column, value_column = "Date", "Close"
    elif {"DATE", "VIXCLS"}.issubset(raw.columns):
        date_column, value_column = "DATE", "VIXCLS"
    else:
        raise ValueError("Accepted schemas are exactly Date,Close or DATE,VIXCLS.")

    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw[date_column], errors="coerce"),
            "VIX": pd.to_numeric(raw[value_column], errors="coerce"),
        }
    )
    valid = data["Date"].notna() & data["VIX"].notna() & data["VIX"].gt(0)
    dropped_rows = int((~valid).sum())
    data = data.loc[valid].sort_values("Date").reset_index(drop=True)
    if data.empty:
        raise ValueError("The approved CSV contains no valid positive VIX observations.")
    duplicate_dates = data.loc[data["Date"].duplicated(keep=False), "Date"]
    if not duplicate_dates.empty:
        raise ValueError(f"Duplicate VIX dates found: {duplicate_dates.head().tolist()}")

    print(f"VIX source: {path}")
    print(f"Dropped rows: {dropped_rows}")
    print(f"Range: {data['Date'].iloc[0].date()} to {data['Date'].iloc[-1].date()} | n={len(data)}")
    return data


vix_data = load_vix_csv(locate_approved_csv(LOCAL_VIX_PATHS, "VIX"))
dates = vix_data["Date"].to_numpy()
vix_level = vix_data["VIX"].to_numpy(dtype=np.float64)
log_vix = np.log(vix_level)
n_obs = len(vix_data)


# Split
validation_start = int(np.floor(0.60 * n_obs))
test_start = int(np.floor(0.80 * n_obs))
minimum_origin = max(MAX_WINDOW - 1, HAR_MONTH - 1)
train_origins = np.arange(minimum_origin, validation_start - MAX_HORIZON, dtype=int)
validation_origins = np.arange(max(validation_start, minimum_origin), test_start - MAX_HORIZON, dtype=int)
refit_origins = np.arange(minimum_origin, test_start - MAX_HORIZON, dtype=int)
test_origins = np.arange(max(test_start, minimum_origin), n_obs - MAX_HORIZON, dtype=int)
if min(map(len, (train_origins, validation_origins, refit_origins, test_origins))) == 0:
    raise ValueError("The cleaned VIX history is too short for the fixed 60/20/20 protocol.")
assert train_origins[-1] + MAX_HORIZON < validation_start
assert validation_origins[-1] + MAX_HORIZON < test_start
assert refit_origins[-1] + MAX_HORIZON < test_start
assert test_origins[-1] + MAX_HORIZON < n_obs
print(f"Validation origins: {len(validation_origins)} | Test origins: {len(test_origins)}")


# Contexts
def build_contexts(values: np.ndarray, origins: np.ndarray, window_size: int) -> np.ndarray:
    indices = origins[:, None] + np.arange(-window_size + 1, 1)
    contexts = values[indices].astype(np.float32, copy=False)
    if indices.min() < 0 or not np.isfinite(contexts).all():
        raise ValueError("Invalid forecasting contexts.")
    return contexts


def target_matrix(values: np.ndarray, origins: np.ndarray) -> np.ndarray:
    return np.column_stack([values[origins + horizon] for horizon in HORIZONS])


def safe_exp(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, -20.0, 20.0)
    clip_count = int(np.count_nonzero(clipped != values))
    if clip_count:
        warnings.warn(f"Clipped {clip_count} log forecasts to [-20, 20] before exponentiation.", RuntimeWarning)
    return np.exp(clipped)


def check_quantiles(values: np.ndarray, expected_rows: int) -> np.ndarray:
    quantiles = np.asarray(values, dtype=np.float64)
    expected_shape = (expected_rows, MAX_HORIZON, len(QUANTILE_LEVELS))
    if quantiles.shape != expected_shape or not np.isfinite(quantiles).all():
        raise ValueError(f"Invalid quantile tensor {quantiles.shape}; expected {expected_shape}.")
    return np.sort(quantiles, axis=-1) if np.any(np.diff(quantiles, axis=-1) < 0) else quantiles


def check_selected_quantiles(values: np.ndarray) -> np.ndarray:
    expected_shape = (len(values), len(HORIZONS), len(QUANTILE_LEVELS))
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError(f"Invalid selected quantile tensor: {values.shape}")
    return np.sort(values, axis=-1) if np.any(np.diff(values, axis=-1) < 0) else values


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
        log_forecasts = np.empty(len(origins), dtype=np.float64)
        for position, origin in enumerate(origins):
            training = np.arange(HAR_MONTH - 1, origin - horizon + 1)
            x_train = features[training]
            y_train = values[training + horizon]
            finite = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)
            if finite.sum() < 60:
                raise ValueError(f"Only {finite.sum()} valid HAR rows are available at origin {origin}.")
            fit = sm.OLS(y_train[finite], x_train[finite]).fit()
            if not np.isfinite(fit.params).all():
                raise FloatingPointError("HAR-OLS returned non-finite coefficients.")
            log_forecasts[position] = float(features[origin] @ fit.params)
        forecasts[horizon] = safe_exp(log_forecasts)
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
    lag = min(max(0, int(np.floor(n ** (1 / 3)))), n - 1)
    centered = differential - differential.mean()
    long_run_variance = float(centered @ centered / n)
    for current_lag in range(1, lag + 1):
        covariance = float(centered[current_lag:] @ centered[:-current_lag] / n)
        long_run_variance += 2.0 * (1.0 - current_lag / (lag + 1.0)) * covariance
    if long_run_variance <= np.finfo(float).eps:
        return np.nan
    statistic = differential.mean() / math.sqrt(long_run_variance / n)
    statistic *= math.sqrt(max((n + 1.0 - 2.0) / n, 0.0))
    return float(2.0 * student_t.sf(abs(statistic), df=n - 1))


def bootstrap_interval(loss_differential: np.ndarray) -> tuple[float, float]:
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
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([differential[start : start + block_length] for start in selected])[:n]
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
    model_error = np.abs(actual - model_forecast)
    har_error = np.abs(actual - har_forecast)
    differential = model_error - har_error
    lower, upper = bootstrap_interval(differential)
    model_mae = float(model_error.mean())
    har_mae = float(har_error.mean())
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
from chronos import Chronos2Pipeline


class LoRALinear(torch.nn.Module):
    def __init__(self, base_layer: torch.nn.Linear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.dropout = torch.nn.Dropout(LORA_DROPOUT)
        self.lora_a = torch.nn.Linear(base_layer.in_features, LORA_R, bias=False)
        self.lora_b = torch.nn.Linear(LORA_R, base_layer.out_features, bias=False)
        self.scale = LORA_ALPHA / LORA_R
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b.weight)
        self.base_layer.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base_layer(inputs) + self.scale * self.lora_b(self.lora_a(self.dropout(inputs)))


def add_lora_adapters(model: torch.nn.Module) -> int:
    model.requires_grad_(False)
    replacements = []
    for name, module in list(model.named_modules()):
        leaf = name.rsplit(".", 1)[-1]
        is_attention = ".self_attention." in name and leaf in LORA_TARGETS
        is_output = name.endswith("output_patch_embedding.output_layer")
        if not (is_attention or is_output) or not isinstance(module, torch.nn.Linear):
            continue
        parent_name, _, child_name = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child_name, LoRALinear(module))
        replacements.append(name)
    found = {name.rsplit(".", 1)[-1] for name in replacements if ".self_attention." in name}
    if found != LORA_TARGETS or not any(name.endswith("output_patch_embedding.output_layer") for name in replacements):
        raise RuntimeError(f"Incomplete Chronos-2 LoRA targets: {replacements[:10]}")
    return len(replacements)


def verify_lora(model: torch.nn.Module) -> None:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Only LoRA parameters must be trainable.")
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"LoRA trainable parameters: {trainable_count:,}/{total_count:,}")


def build_trainable_bundle() -> dict[str, object]:
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, device_map="cuda", torch_dtype=torch.float32
    )
    model = pipeline.model
    replacement_count = add_lora_adapters(model)
    model.to(DEVICE)
    pipeline.model = model
    verify_lora(model)
    print(f"LoRA projections replaced: {replacement_count}")
    return {"pipeline": pipeline, "model": model}


def make_loader(contexts: np.ndarray, targets: np.ndarray, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(contexts), torch.from_numpy(targets))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)


def supervised_arrays(origins: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    contexts = build_contexts(log_vix, origins, window_size)
    offsets = np.arange(1, MAX_HORIZON + 1)
    targets = log_vix[origins[:, None] + offsets].astype(np.float32)
    return contexts, targets


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    trainable_names = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if set(state) != trainable_names:
        raise RuntimeError("The saved Chronos-2 adapter state does not match the trainable parameters.")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))


def training_loss(bundle: dict[str, object], batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    context, target = (tensor.to(DEVICE) for tensor in batch)
    model = bundle["model"]
    output_patch_size = int(model.chronos_config.output_patch_size)
    return model(
        context=context,
        group_ids=torch.arange(len(context), device=DEVICE),
        num_output_patches=math.ceil(MAX_HORIZON / output_patch_size),
        future_target=target,
    ).loss


def predict_bundle_quantiles(bundle: dict[str, object], contexts: np.ndarray) -> np.ndarray:
    model = bundle["model"]
    pipeline = bundle["pipeline"]
    model.eval()
    batches = []
    for start in range(0, len(contexts), INFERENCE_BATCH_SIZE):
        context = torch.from_numpy(contexts[start : start + INFERENCE_BATCH_SIZE]).unsqueeze(1)
        with torch.inference_mode():
            quantile_list, _ = pipeline.predict_quantiles(
                inputs=context,
                prediction_length=MAX_HORIZON,
                quantile_levels=QUANTILE_LEVELS,
                batch_size=len(context),
                context_length=contexts.shape[1],
                cross_learning=False,
            )
        expected_shape = (1, MAX_HORIZON, len(QUANTILE_LEVELS))
        if not isinstance(quantile_list, list) or len(quantile_list) != len(context):
            raise TypeError("Expected one documented Chronos-2 tensor per input series.")
        if any(tensor.shape != expected_shape for tensor in quantile_list):
            raise ValueError(f"Unexpected Chronos-2 quantile shapes: {[tuple(x.shape) for x in quantile_list[:3]]}")
        batches.append(torch.stack([tensor[0] for tensor in quantile_list]).detach().float().cpu().numpy())
    return np.concatenate(batches)


def optimizer_updates(
    bundle: dict[str, object],
    loader: DataLoader,
    maximum_updates: int,
    validation_callback=None,
    snapshot_updates: set[int] | None = None,
) -> tuple[dict[int, dict[str, torch.Tensor]], int]:
    model = bundle["model"]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    iterator = iter(loader)
    snapshots = {}
    completed = accumulated = 0
    optimizer.zero_grad(set_to_none=True)
    while completed < maximum_updates:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        model.train()
        loss = training_loss(bundle, batch)
        if not torch.isfinite(loss):
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
            snapshots[completed] = trainable_state(model)
        if validation_callback and completed % VALIDATION_EVERY_UPDATES == 0 and validation_callback(completed):
            break
    del optimizer
    return snapshots, completed


def select_updates(window_size: int, seed: int) -> dict[int, int]:
    set_seed(seed)
    bundle = build_trainable_bundle()
    contexts, targets = supervised_arrays(train_origins, window_size)
    loader = make_loader(contexts, targets, seed)
    validation_contexts = build_contexts(log_vix, validation_origins, window_size)
    best_mae = {horizon: np.inf for horizon in HORIZONS}
    best_update = {horizon: -1 for horizon in HORIZONS}
    checks_without_improvement = 0

    def validate(update: int) -> bool:
        nonlocal checks_without_improvement
        quantiles = check_quantiles(predict_bundle_quantiles(bundle, validation_contexts), len(validation_origins))
        level_quantiles = safe_exp(quantiles[:, HORIZON_STEPS, :])
        improved = False
        for index, horizon in enumerate(HORIZONS):
            mae = float(np.mean(np.abs(validation_actual[:, index] - level_quantiles[:, index, 1])))
            print(f"W={window_size} seed={seed} update={update} h={horizon} validation MAE={mae:.6f}")
            if update >= MIN_UPDATES and mae < best_mae[horizon] - 1e-10:
                best_mae[horizon] = mae
                best_update[horizon] = update
                improved = True
        checks_without_improvement = 0 if improved else checks_without_improvement + 1
        return update >= MIN_UPDATES and checks_without_improvement >= EARLY_STOPPING_PATIENCE

    optimizer_updates(bundle, loader, MAX_UPDATES, validation_callback=validate)
    if any(update < 0 for update in best_update.values()):
        raise RuntimeError(f"No eligible update selected for W={window_size}, seed={seed}.")
    del bundle, contexts, targets, loader, validation_contexts
    gc.collect()
    torch.cuda.empty_cache()
    return best_update


def refit_and_forecast(window_size: int, seed: int, selected_updates: dict[int, int]) -> np.ndarray:
    set_seed(seed)
    bundle = build_trainable_bundle()
    contexts, targets = supervised_arrays(refit_origins, window_size)
    loader = make_loader(contexts, targets, seed)
    required_updates = set(selected_updates.values())
    snapshots, completed = optimizer_updates(bundle, loader, max(required_updates), snapshot_updates=required_updates)
    if completed != max(required_updates) or set(snapshots) != required_updates:
        raise RuntimeError("Refit did not preserve every required optimizer-update state.")
    test_contexts = build_contexts(log_vix, test_origins, window_size)
    selected_quantiles = []
    for index, horizon in enumerate(HORIZONS):
        restore_trainable_state(bundle["model"], snapshots[selected_updates[horizon]])
        quantiles = check_quantiles(predict_bundle_quantiles(bundle, test_contexts), len(test_origins))
        selected_quantiles.append(quantiles[:, HORIZON_STEPS[index], :])
    result = check_selected_quantiles(np.stack(selected_quantiles, axis=1))
    del bundle, contexts, targets, loader, test_contexts, snapshots
    gc.collect()
    torch.cuda.empty_cache()
    return result


# Window loop, validation selection, causal refit and test
validation_actual = target_matrix(vix_level, validation_origins)
test_actual = target_matrix(vix_level, test_origins)
forecast_store = {}
seed_records = []

for window_size in WINDOW_SIZES:
    print(f"Fine-tuning W={window_size}")
    test_by_seed = []
    for seed in SEEDS:
        selected_updates = select_updates(window_size, seed)
        test_by_seed.append(refit_and_forecast(window_size, seed, selected_updates))

    test_seed_matrix = np.stack(test_by_seed)
    ensemble_log = check_selected_quantiles(np.median(test_seed_matrix, axis=0))
    forecast_store[window_size] = safe_exp(ensemble_log)
    seed_levels = safe_exp(test_seed_matrix[..., 1])
    for horizon_index, horizon in enumerate(HORIZONS):
        for seed_index, seed in enumerate(SEEDS):
            seed_records.append(
                {
                    "window_size": window_size,
                    "horizon": horizon,
                    "seed": seed,
                    "seed_MAE": float(
                        np.mean(np.abs(test_actual[:, horizon_index] - seed_levels[seed_index, :, horizon_index]))
                    ),
                }
            )
    del test_by_seed, test_seed_matrix, seed_levels
    gc.collect()
    torch.cuda.empty_cache()


# Metrics and summary rows
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
rows = []
for window_size in WINDOW_SIZES:
    for horizon_index, horizon in enumerate(HORIZONS):
        positions = np.arange(0, len(test_origins), horizon)
        rows.append(
            comparison_row(
                window_size,
                horizon,
                test_actual[positions, horizon_index],
                forecast_store[window_size][positions, horizon_index, 1],
                test_har[horizon][positions],
            )
        )
results = pd.DataFrame(rows, columns=OUTPUT_COLUMNS).sort_values(["window_size", "horizon"]).reset_index(drop=True)


# CSV
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
for axis, horizon in zip(axes, HORIZONS):
    horizon_rows = primary_har_ols[primary_har_ols["horizon"] == horizon].sort_values("window_size")
    axis.plot(horizon_rows["window_size"], horizon_rows["model_MAE"], marker="o", label=MODEL_FAMILY)
    axis.plot(horizon_rows["window_size"], horizon_rows["har_MAE"], marker="s", label="HAR-OLS")
    axis.set_title(f"MAE, h={horizon}")
    axis.set_xlabel("Window")
    axis.grid(alpha=0.3)
axes[0].legend()
fig.tight_layout()
plt.show()
plt.close(fig)

delta_matrix = primary_har_ols.pivot(index="window_size", columns="horizon", values="delta_MAE").reindex(
    index=WINDOW_SIZES, columns=HORIZONS
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
    positions = np.arange(0, len(test_origins), horizon)
    target_dates = pd.to_datetime(dates[test_origins[positions] + horizon])
    observed = test_actual[positions, horizon_index]
    for axis, window_size in zip(axes, WINDOW_SIZES):
        quantiles = forecast_store[window_size][positions, horizon_index]
        axis.plot(target_dates, observed, color="black", linewidth=1, label="Observed")
        axis.plot(target_dates, quantiles[:, 1], color="tab:blue", linewidth=1, label="q50")
        axis.plot(target_dates, test_har[horizon][positions], color="tab:orange", linewidth=1, label="HAR-OLS")
        axis.fill_between(target_dates, quantiles[:, 0], quantiles[:, 2], color="tab:blue", alpha=0.18, label="q10-q90")
        axis.set_title(f"W={window_size}")
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=4)
    fig.suptitle(f"Non-overlapping test paths, horizon {horizon}")
    fig.tight_layout()
    plt.show()
    plt.close(fig)

seed_plot_table = pd.DataFrame(seed_records)
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
assert all(values.shape == (len(test_origins), len(HORIZONS), 3) for values in forecast_store.values())
assert all(np.isfinite(values).all() for values in forecast_store.values())
assert OUTPUT_FILENAME.count(".csv") == 1 and "_w" not in OUTPUT_FILENAME
print("All protocol assertions passed.")

"""### Classifier"""

# Commented out IPython magic to ensure Python compatibility.

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q --upgrade-strategy only-if-needed "chronos-forecasting==2.3.1"

