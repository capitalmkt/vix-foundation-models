"""
03. Chronos-Bolt - Direction Classification (multivariate)

Model family: chronos_bolt
Adaptation: classifier
Information set: VIX+VVIX
Covariate method: dual_encoder_vix_vvix
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
import copy
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
import torch
from IPython.display import display
from scipy.stats import t as student_t
import torch.nn.functional as F
from chronos import BaseChronosPipeline
from torch.utils.data import DataLoader, TensorDataset

# Configuration
MODEL_ID = "amazon/chronos-bolt-base"
MODEL_REVISION = "5d9f166d69f47aef3401367a7b842e78fe97b121"
COVARIATE_METHOD = "dual_encoder_vix_vvix"
INFORMATION_SET = "VIX+VVIX"
WINDOW_SIZES = [16, 32, 64, 128, 256]
HORIZONS = [1, 5, 22]
MAX_WINDOW = max(WINDOW_SIZES)
MAX_HORIZON = max(HORIZONS)
HAR_WEEK = 5
HAR_MONTH = 22
SEEDS = [11, 22, 33, 44, 55]
BOOTSTRAP_REPLICATIONS = 5000
BOOTSTRAP_SEED = 2026
OUTPUT_FILENAME = "classifier_chronos_bolt_multivariate.csv"
LOCAL_VIX_PATHS = [REPO_DATA_DIR / "VIX.csv", Path("/content/VIX.csv"), Path("/content/VIXCLS.csv"), Path("/content/vix.csv")]
LOCAL_VVIX_PATHS = [REPO_DATA_DIR / "VVIX.csv", Path("/content/VVIX.csv"), Path("/content/VVIXCLS.csv"), Path("/content/vvix.csv")]
RESULTS_DIRECTORY = resolve_results_directory()
enable_figure_autosave(RESULTS_DIRECTORY, Path(OUTPUT_FILENAME).stem)
DEVICE = torch.device("cuda")

if not torch.cuda.is_available():
    raise RuntimeError("Select a T4 GPU runtime in Google Colab before running this cell.")

# Reproducibility


set_seed(SEEDS[0])


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
validation_start = int(np.floor(0.60 * n_obs))
test_start = int(np.floor(0.80 * n_obs))
minimum_origin = max(MAX_WINDOW - 1, HAR_MONTH - 1)
train_origins = np.arange(minimum_origin, validation_start - MAX_HORIZON, dtype=int)
validation_origins = np.arange(max(validation_start, minimum_origin), test_start - MAX_HORIZON, dtype=int)
refit_origins = np.arange(minimum_origin, test_start - MAX_HORIZON, dtype=int)
test_origins = np.arange(max(test_start, minimum_origin), n_obs - MAX_HORIZON, dtype=int)
if min(map(len, [train_origins, validation_origins, refit_origins, test_origins])) == 0:
    raise ValueError("The aligned history is too short for the fixed 60/20/20 protocol.")
assert train_origins[-1] + MAX_HORIZON < validation_start
assert validation_origins[-1] + MAX_HORIZON < test_start
assert refit_origins[-1] + MAX_HORIZON < test_start
assert test_origins[-1] + MAX_HORIZON < n_obs
print(f"Train/validation/refit/test origins: {len(train_origins)}/{len(validation_origins)}/{len(refit_origins)}/{len(test_origins)}")


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


har_ols_level = np.exp(
    np.clip(expanding_direct_ols(har_design(log_vix, None), log_vix, test_origins, "HAR-OLS"), -20.0, 20.0)
)
har_x_ols_level = np.exp(
    np.clip(expanding_direct_ols(har_design(log_vix, log_vvix), log_vix, test_origins, "HAR-X-OLS"), -20.0, 20.0)
)


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


# Family-specific dual encoder
MODEL_DISPLAY_NAME = "Chronos-Bolt+VVIX ensemble"
# Local LoRA injection avoids text-embedding assumptions in time-series backbones
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


class LoRALinear(torch.nn.Module):
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
        return self.base_layer(inputs) + self.scale * self.lora_b(self.lora_a(self.lora_dropout(inputs)))


def add_lora_adapters(model: torch.nn.Module) -> list[str]:
    model.requires_grad_(False)
    replacements = []
    for full_name, module in list(model.named_modules()):
        leaf_name = full_name.rsplit(".", 1)[-1]
        if leaf_name not in {"q", "v"} or not isinstance(module, torch.nn.Linear):
            continue
        parent_name, _, child_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module))
        replacements.append(full_name)
    if not replacements:
        raise RuntimeError("The fixed LoRA selector did not match any real linear module.")
    return replacements


def verify_lora(model: torch.nn.Module) -> None:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    lora = [name for name in trainable if "lora_" in name]
    head = [name for name in trainable if "classifier_head" in name]
    unexpected = [name for name in trainable if "lora_" not in name and "classifier_head" not in name]
    if not lora or not head or unexpected:
        raise RuntimeError(
            f"Invalid trainable state: LoRA={len(lora)}, head={len(head)}, unexpected={unexpected[:5]}"
        )
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {trainable_count:,}/{total_count:,} | LoRA tensors={len(lora)} | head tensors={len(head)}")

print(
    f"Python {platform.python_version()} | PyTorch {torch.__version__} | "
    f"chronos-forecasting {importlib.metadata.version('chronos-forecasting')} | GPU {torch.cuda.get_device_name(0)}"
)
pristine_pipeline = BaseChronosPipeline.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, device_map="cpu", torch_dtype=torch.float32
)


class BoltMultivariateDirectionClassifier(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier_head = torch.nn.Linear(2 * hidden_size, len(HORIZONS) * N_CLASSES)

    def encode_series(self, context: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(context)
        hidden, _, _, attention_mask = self.backbone.encode(context=context, mask=mask)
        weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(self, vix_context: torch.Tensor, vvix_context: torch.Tensor) -> torch.Tensor:
        vix_embedding = self.encode_series(vix_context)
        vvix_embedding = self.encode_series(vvix_context)
        joint = torch.cat([vix_embedding, vvix_embedding], dim=-1)
        return self.classifier_head(joint.float()).view(len(vix_context), len(HORIZONS), N_CLASSES)


def build_direction_classifier() -> BoltMultivariateDirectionClassifier:
    backbone = copy.deepcopy(pristine_pipeline.model)
    replacements = add_lora_adapters(backbone)
    backbone.to(DEVICE)
    model = BoltMultivariateDirectionClassifier(backbone, int(backbone.config.d_model)).to(DEVICE)
    verify_lora(model)
    print(f"LoRA projections replaced: {len(replacements)}")
    return model

# Labels and classifier metrics
DOWN_CLASS = 0
NEUTRAL_CLASS = 1
UP_CLASS = 2
N_CLASSES = 3
MOVE_THRESHOLD = 0.01


def classify_vix_change(change: np.ndarray) -> np.ndarray:
    return np.select(
        [change < -MOVE_THRESHOLD, change > MOVE_THRESHOLD],
        [DOWN_CLASS, UP_CLASS],
        default=NEUTRAL_CLASS,
    ).astype(np.int64)


def label_matrix(origins: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [classify_vix_change(vix_level[origins + horizon] / vix_level[origins] - 1.0) for horizon in HORIZONS]
    )


train_labels = label_matrix(train_origins)
validation_labels = label_matrix(validation_origins)
refit_labels = label_matrix(refit_origins)
test_labels = label_matrix(test_origins)
np.testing.assert_array_equal(
    classify_vix_change(np.array([-0.011, -0.010, 0.0, 0.010, 0.011])),
    np.array([DOWN_CLASS, NEUTRAL_CLASS, NEUTRAL_CLASS, NEUTRAL_CLASS, UP_CLASS]),
)


def multiclass_metrics(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(actual), N_CLASSES) or not np.isfinite(probabilities).all():
        raise ValueError(f"Invalid probability tensor: {probabilities.shape}")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(confusion, (actual, predicted), 1)
    support = confusion.sum(axis=1)
    present = support > 0
    recall = np.divide(np.diag(confusion), support, out=np.zeros(N_CLASSES, dtype=float), where=support > 0)
    predicted_support = confusion.sum(axis=0)
    precision = np.divide(np.diag(confusion), predicted_support, out=np.zeros(N_CLASSES, dtype=float), where=predicted_support > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros(N_CLASSES, dtype=float), where=(precision + recall) > 0)
    true_probability = np.clip(probabilities[np.arange(len(actual)), actual], 1e-12, 1.0)
    return {
        "balanced_accuracy": float(np.mean(recall[present])),
        "macro_f1": float(np.mean(f1[present])),
        "accuracy": float(np.mean(predicted == actual)),
        "log_loss": float(-np.mean(np.log(true_probability))),
    }


def deterministic_probabilities(classes: np.ndarray) -> np.ndarray:
    return np.eye(N_CLASSES, dtype=np.float64)[classes]


test_har_classes = {
    "HAR-OLS": np.column_stack(
        [classify_vix_change(har_ols_level[:, index] / vix_level[test_origins] - 1.0) for index in range(len(HORIZONS))]
    ),
    "HAR-X-OLS": np.column_stack(
        [classify_vix_change(har_x_ols_level[:, index] / vix_level[test_origins] - 1.0) for index in range(len(HORIZONS))]
    ),
}


# Optimizer-update protocol
LEARNING_RATE = 1e-4
TRAIN_BATCH_SIZE = 32
INFERENCE_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 2
MIN_UPDATES = 100
VALIDATION_EVERY_UPDATES = 50
MAX_UPDATES = 500
EARLY_STOPPING_PATIENCE = 3
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0


def make_classifier_loader(vix_contexts: np.ndarray, vvix_contexts: np.ndarray, labels: np.ndarray, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(vix_contexts), torch.from_numpy(vvix_contexts), torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def restore_trainable_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    if set(state) != {name for name, parameter in parameters.items() if parameter.requires_grad}:
        raise RuntimeError("Saved classifier state does not match current trainable parameters.")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))


def classifier_probabilities(model: torch.nn.Module, vix_contexts: np.ndarray, vvix_contexts: np.ndarray) -> np.ndarray:
    model.eval()
    batches = []
    start = 0
    batch_size = min(INFERENCE_BATCH_SIZE, len(vix_contexts))
    while start < len(vix_contexts):
        stop = min(start + batch_size, len(vix_contexts))
        vix_batch = torch.as_tensor(vix_contexts[start:stop], dtype=torch.float32, device=DEVICE)
        vvix_batch = torch.as_tensor(vvix_contexts[start:stop], dtype=torch.float32, device=DEVICE)
        try:
            with torch.inference_mode():
                logits = model(vix_batch, vvix_batch)
            expected = (stop - start, len(HORIZONS), N_CLASSES)
            if tuple(logits.shape) != expected or not torch.isfinite(logits).all():
                raise ValueError(f"Unexpected classifier logits: {tuple(logits.shape)}, expected {expected}.")
            batches.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
            start = stop
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            warnings.warn(f"CUDA OOM: classifier inference batch reduced to {batch_size}.", RuntimeWarning)
    probabilities = np.concatenate(batches)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-5)
    return probabilities


def run_classifier_updates(
    model: torch.nn.Module,
    loader: DataLoader,
    maximum_updates: int,
    validation_callback: Any | None = None,
    snapshot_updates: set[int] | None = None,
) -> tuple[dict[int, dict[str, torch.Tensor]], int]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    iterator = iter(loader)
    completed = accumulated = 0
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    optimizer.zero_grad(set_to_none=True)
    while completed < maximum_updates:
        try:
            vix_batch, vvix_batch, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            vix_batch, vvix_batch, labels = next(iterator)
        vix_batch, vvix_batch, labels = vix_batch.to(DEVICE), vvix_batch.to(DEVICE), labels.to(DEVICE)
        model.train()
        logits = model(vix_batch, vvix_batch)
        loss = F.cross_entropy(logits.reshape(-1, N_CLASSES), labels.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError("Classifier loss is not finite.")
        (loss / GRADIENT_ACCUMULATION_STEPS).backward()
        accumulated += 1
        if accumulated < GRADIENT_ACCUMULATION_STEPS:
            continue
        torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed += 1
        accumulated = 0
        if snapshot_updates and completed in snapshot_updates:
            snapshots[completed] = trainable_state(model)
        if validation_callback and completed % VALIDATION_EVERY_UPDATES == 0:
            if validation_callback(completed):
                break
    del optimizer
    return snapshots, completed


def select_classifier_updates(window_size: int, seed: int) -> dict[int, int]:
    set_seed(seed)
    model = build_direction_classifier()
    train_vix, train_vvix = build_aligned_contexts(log_vix, log_vvix, train_origins, window_size)
    validation_vix, validation_vvix = build_aligned_contexts(log_vix, log_vvix, validation_origins, window_size)
    loader = make_classifier_loader(train_vix, train_vvix, train_labels, seed)
    best_score = {horizon: (-np.inf, -np.inf, -np.inf) for horizon in HORIZONS}
    best_update = {horizon: -1 for horizon in HORIZONS}
    checks_without_improvement = 0

    def validate(update: int) -> bool:
        nonlocal checks_without_improvement
        probabilities = classifier_probabilities(model, validation_vix, validation_vvix)
        improved = False
        for index, horizon in enumerate(HORIZONS):
            metrics = multiclass_metrics(validation_labels[:, index], probabilities[:, index])
            candidate = (metrics["balanced_accuracy"], -metrics["log_loss"], -update)
            if update >= MIN_UPDATES and candidate > best_score[horizon]:
                best_score[horizon] = candidate
                best_update[horizon] = update
                improved = True
            print(
                f"W={window_size} seed={seed} update={update} h={horizon} "
                f"balanced_accuracy={metrics['balanced_accuracy']:.6f} log_loss={metrics['log_loss']:.6f}"
            )
        checks_without_improvement = 0 if improved else checks_without_improvement + 1
        return update >= MIN_UPDATES and checks_without_improvement >= EARLY_STOPPING_PATIENCE

    run_classifier_updates(model, loader, MAX_UPDATES, validation_callback=validate)
    if any(update < 0 for update in best_update.values()):
        raise RuntimeError(f"No eligible classifier update selected for W={window_size}, seed={seed}.")
    del model, train_vix, train_vvix, validation_vix, validation_vvix, loader
    gc.collect()
    torch.cuda.empty_cache()
    return best_update


def refit_classifier(window_size: int, seed: int, selected_updates: dict[int, int]) -> np.ndarray:
    set_seed(seed)
    model = build_direction_classifier()
    refit_vix, refit_vvix = build_aligned_contexts(log_vix, log_vvix, refit_origins, window_size)
    loader = make_classifier_loader(refit_vix, refit_vvix, refit_labels, seed)
    required = set(selected_updates.values())
    snapshots, completed = run_classifier_updates(model, loader, max(required), snapshot_updates=required)
    if completed != max(required) or set(snapshots) != required:
        raise RuntimeError("Classifier refit did not preserve every selected update state.")
    test_vix, test_vvix = build_aligned_contexts(log_vix, log_vvix, test_origins, window_size)
    selected = []
    for index, horizon in enumerate(HORIZONS):
        restore_trainable_state(model, snapshots[selected_updates[horizon]])
        selected.append(classifier_probabilities(model, test_vix, test_vvix)[:, index])
    probabilities = np.stack(selected, axis=1)
    del model, refit_vix, refit_vvix, test_vix, test_vvix, loader, snapshots
    gc.collect()
    torch.cuda.empty_cache()
    return probabilities


# Validation selection, causal refit and ensemble
classifier_store: dict[int, np.ndarray] = {}
for window_size in WINDOW_SIZES:
    print(f"Training multivariate classifier W={window_size}")
    test_by_seed = []
    for seed in SEEDS:
        selected_updates = select_classifier_updates(window_size, seed)
        test_probability = refit_classifier(window_size, seed, selected_updates)
        test_by_seed.append(test_probability)
    test_seed_tensor = np.stack(test_by_seed)
    test_ensemble = test_seed_tensor.mean(axis=0)
    np.testing.assert_allclose(test_ensemble.sum(axis=-1), 1.0, atol=1e-5)
    for horizon_index, horizon in enumerate(HORIZONS):
        positions = np.arange(0, len(test_origins), horizon, dtype=int)
        seed_scores = np.array(
            [multiclass_metrics(test_labels[positions, horizon_index], tensor[positions, horizon_index])["balanced_accuracy"] for tensor in test_seed_tensor]
        )
        print(
            f"W={window_size} h={horizon} seed balanced accuracy: "
            f"mean={seed_scores.mean():.4f}, std={seed_scores.std(ddof=1):.4f}, "
            f"min={seed_scores.min():.4f}, max={seed_scores.max():.4f}"
        )
    classifier_store[window_size] = test_ensemble
    del test_by_seed, test_seed_tensor
    gc.collect()
    torch.cuda.empty_cache()

# Compact classifier results
CLASSIFIER_OUTPUT_COLUMNS = [
    "window_size", "horizon", "benchmark", "n_obs",
    "model_balanced_accuracy", "har_balanced_accuracy", "delta_balanced_accuracy",
    "model_macro_f1", "har_macro_f1", "delta_macro_f1",
    "model_accuracy_pct", "har_accuracy_pct", "delta_accuracy_pct_points",
    "dm_p_value_unadjusted", "accuracy_delta_ci_lower_pct_points", "accuracy_delta_ci_upper_pct_points",
]
rows: list[dict[str, Any]] = []
for window_size in WINDOW_SIZES:
    probabilities = classifier_store[window_size]
    for horizon_index, horizon in enumerate(HORIZONS):
        positions = np.arange(0, len(test_origins), horizon, dtype=int)
        actual = test_labels[positions, horizon_index]
        model_probability = probabilities[positions, horizon_index]
        model_metrics = multiclass_metrics(actual, model_probability)
        model_class = model_probability.argmax(axis=1)
        for benchmark_name, benchmark_classes in test_har_classes.items():
            benchmark_class = benchmark_classes[positions, horizon_index]
            benchmark_metrics = multiclass_metrics(actual, deterministic_probabilities(benchmark_class))
            model_loss = (model_class != actual).astype(float)
            benchmark_loss = (benchmark_class != actual).astype(float)
            differential = model_loss - benchmark_loss
            loss_lower, loss_upper = moving_block_bootstrap(differential)
            rows.append(
                {
                    "window_size": window_size,
                    "horizon": horizon,
                    "benchmark": benchmark_name,
                    "n_obs": len(actual),
                    "model_balanced_accuracy": model_metrics["balanced_accuracy"],
                    "har_balanced_accuracy": benchmark_metrics["balanced_accuracy"],
                    "delta_balanced_accuracy": model_metrics["balanced_accuracy"] - benchmark_metrics["balanced_accuracy"],
                    "model_macro_f1": model_metrics["macro_f1"],
                    "har_macro_f1": benchmark_metrics["macro_f1"],
                    "delta_macro_f1": model_metrics["macro_f1"] - benchmark_metrics["macro_f1"],
                    "model_accuracy_pct": 100.0 * model_metrics["accuracy"],
                    "har_accuracy_pct": 100.0 * benchmark_metrics["accuracy"],
                    "delta_accuracy_pct_points": 100.0 * (model_metrics["accuracy"] - benchmark_metrics["accuracy"]),
                    "dm_p_value_unadjusted": dm_p_value(differential),
                    "accuracy_delta_ci_lower_pct_points": -100.0 * loss_upper,
                    "accuracy_delta_ci_upper_pct_points": -100.0 * loss_lower,
                }
            )

results = pd.DataFrame(rows, columns=CLASSIFIER_OUTPUT_COLUMNS).sort_values(
    ["window_size", "horizon", "benchmark"], kind="stable"
).reset_index(drop=True)
assert list(results.columns) == CLASSIFIER_OUTPUT_COLUMNS
assert len(results) == len(WINDOW_SIZES) * len(HORIZONS) * 2
assert set(results["window_size"].astype(int)) == set(WINDOW_SIZES)
assert set(results["horizon"].astype(int)) == set(HORIZONS)
assert set(results["benchmark"]) == {"HAR-OLS", "HAR-X-OLS"}
assert not results.duplicated(["window_size", "horizon", "benchmark"]).any()
np.testing.assert_allclose(results["delta_balanced_accuracy"], results["model_balanced_accuracy"] - results["har_balanced_accuracy"])
np.testing.assert_allclose(results["delta_macro_f1"], results["model_macro_f1"] - results["har_macro_f1"])
np.testing.assert_allclose(results["delta_accuracy_pct_points"], results["model_accuracy_pct"] - results["har_accuracy_pct"])

RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
csv_path = RESULTS_DIRECTORY / OUTPUT_FILENAME
results.to_csv(csv_path, index=False)
print(f"Saved compact results CSV: {csv_path}")
display(results)


# Exactly two figures
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for axis, horizon in zip(axes, HORIZONS):
    current = results[results["horizon"] == horizon]
    model_line = current.drop_duplicates("window_size").sort_values("window_size")
    axis.plot(model_line["window_size"], model_line["model_balanced_accuracy"], marker="o", label=MODEL_DISPLAY_NAME)
    for benchmark_name, marker in [("HAR-OLS", "s"), ("HAR-X-OLS", "^")]:
        line = current[current["benchmark"] == benchmark_name].sort_values("window_size")
        axis.plot(line["window_size"], line["har_balanced_accuracy"], marker=marker, label=benchmark_name)
    axis.set_title(f"Balanced accuracy, h={horizon}")
    axis.set_xlabel("Window")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
axes[0].set_ylabel("Balanced accuracy")
axes[0].legend()
fig.tight_layout()
plt.show()
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
x = np.arange(len(WINDOW_SIZES))
width = 0.25
for axis, horizon in zip(axes, HORIZONS):
    current = results[results["horizon"] == horizon].set_index(["window_size", "benchmark"])
    model_values = [current.loc[(window, "HAR-OLS"), "model_accuracy_pct"] for window in WINDOW_SIZES]
    har_values = [current.loc[(window, "HAR-OLS"), "har_accuracy_pct"] for window in WINDOW_SIZES]
    har_x_values = [current.loc[(window, "HAR-X-OLS"), "har_accuracy_pct"] for window in WINDOW_SIZES]
    bars_model = axis.bar(x - width, model_values, width, label=MODEL_DISPLAY_NAME)
    bars_har = axis.bar(x, har_values, width, label="HAR-OLS")
    bars_har_x = axis.bar(x + width, har_x_values, width, label="HAR-X-OLS")
    for bars in [bars_model, bars_har, bars_har_x]:
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    axis.set_xticks(x, WINDOW_SIZES)
    axis.set_xlabel("Window")
    axis.set_title(f"Accuracy (%), h={horizon}")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend(fontsize=8)
fig.tight_layout()
plt.show()
plt.close(fig)

assert all(classifier_store[window].shape == (len(test_origins), len(HORIZONS), N_CLASSES) for window in WINDOW_SIZES)
assert all(np.isfinite(classifier_store[window]).all() for window in WINDOW_SIZES)
assert results[["model_balanced_accuracy", "har_balanced_accuracy", "model_macro_f1", "har_macro_f1"]].apply(lambda column: column.between(0, 1).all()).all()
assert results[["model_accuracy_pct", "har_accuracy_pct"]].apply(lambda column: column.between(0, 100).all()).all()
assert "row_type" not in results.columns and "covariate_method" not in results.columns
print("All multivariate classifier assertions passed.")

"""## Chronos 2

### Zero Shot
"""

# Commented out IPython magic to ensure Python compatibility.

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q --upgrade-strategy only-if-needed "chronos-forecasting==2.3.1"

