"""
09. Chronos-2 - three-class direction classifier (univariate)

Model family: Chronos-2
Adaptation: three-class direction classifier
Information set: VIX univariate only
Colab layout: run this complete file as one cell on a T4 GPU
Fixed windows: 16, 32, 64, 128, 256. No window selection is performed.
Fixed horizons: 1, 5, 22 trading days.
Labels: DOWN_CLASS=0, NEUTRAL_CLASS=1, UP_CLASS=2 with a fixed 1% level-move threshold.
Model ID: amazon/chronos-2
Checkpoint revision: 29ec3766d36d6f73f0696f85560a422f50e8498c
Training: five fixed seeds, one fixed family-specific LoRA configuration, unweighted cross-entropy.
Selection: validation balanced accuracy, then log loss, then fewer optimizer updates.
Ensemble: arithmetic mean of the five seed probability tensors, followed by argmax.
Benchmark: HAR-OLS using Corsi's daily/weekly/monthly structure on log-VIX with direct horizons.
Output configuration: one compact 15-row table is displayed and saved unchanged to this cell's CSV.
Plot configuration: three balanced-accuracy line panels and one grouped accuracy-percentage bar figure.
Missing-class policy: balanced accuracy and macro-F1 average only classes present in each evaluated subset.
LoRA integration: local linear adapters avoid any dependency on a token-embedding interface.
Live data policy: disabled. An approved VIX CSV is mandatory.

Documentation:
  https://github.com/amazon-science/chronos-forecasting
  https://github.com/amazon-science/chronos-forecasting/blob/main/pyproject.toml
  https://huggingface.co/amazon/chronos-2
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
import math
import random
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn.functional as F
from chronos import Chronos2Pipeline
from IPython.display import display
from scipy.stats import t as student_t
from sklearn.metrics import log_loss, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

# Configuration
MODEL_ID = "amazon/chronos-2"
MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
WINDOW_SIZES = [16, 32, 64, 128, 256]
HORIZONS = [1, 5, 22]
MAX_HORIZON = 22
MAX_WINDOW = 256
HAR_MONTH = 22
HAR_WEEK = 5
SEEDS = [11, 22, 33, 44, 55]
DOWN_CLASS = 0
NEUTRAL_CLASS = 1
UP_CLASS = 2
CLASS_IDS = [DOWN_CLASS, NEUTRAL_CLASS, UP_CLASS]
N_CLASSES = 3
MOVE_THRESHOLD = 0.01
BOOTSTRAP_REPLICATIONS = 5000
BOOTSTRAP_SEED = 2026
OUTPUT_FILENAME = "classifier_chronos_2_univariate.csv"
LOCAL_VIX_PATHS = [REPO_DATA_DIR / "VIX.csv", Path("/content/VIX.csv"), Path("/content/VIXCLS.csv"), Path("/content/vix.csv")]
RESULTS_DIRECTORY = resolve_results_directory()
enable_figure_autosave(RESULTS_DIRECTORY, Path(OUTPUT_FILENAME).stem)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type != "cuda":
    raise RuntimeError("Select a T4 GPU runtime in Google Colab before running this cell.")


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
    data = data.loc[valid].sort_values("Date").reset_index(drop=True)
    if data.empty:
        raise ValueError("The approved CSV contains no valid positive VIX observations.")
    duplicate_dates = data.loc[data["Date"].duplicated(keep=False), "Date"]
    if not duplicate_dates.empty:
        raise ValueError(f"Duplicate VIX dates found: {duplicate_dates.head().tolist()}")
    return data


vix_data = load_vix_csv(locate_approved_csv(LOCAL_VIX_PATHS, "VIX"))
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
if min(map(len, (train_origins, validation_origins, test_origins))) == 0:
    raise ValueError("The cleaned VIX history is too short for the fixed 60/20/20 protocol.")
assert train_origins[-1] + MAX_HORIZON < validation_start
assert validation_origins[-1] + MAX_HORIZON < test_start
assert refit_origins[-1] + MAX_HORIZON < test_start
assert test_origins[-1] + MAX_HORIZON < n_obs


# Contexts and labels
def build_contexts(values: np.ndarray, origins: np.ndarray, window_size: int) -> np.ndarray:
    indices = origins[:, None] + np.arange(-window_size + 1, 1, dtype=int)
    contexts = values[indices].astype(np.float32, copy=False)
    if indices.min() < 0 or not np.isfinite(contexts).all():
        raise ValueError("Invalid classifier contexts.")
    return contexts


def classify_vix_change(change: np.ndarray) -> np.ndarray:
    return np.select(
        [change < -MOVE_THRESHOLD, change > MOVE_THRESHOLD],
        [DOWN_CLASS, UP_CLASS],
        default=NEUTRAL_CLASS,
    ).astype(np.int64)


def label_matrix(origins: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            classify_vix_change(vix_level[origins + horizon] / vix_level[origins] - 1.0)
            for horizon in HORIZONS
        ]
    )


train_labels = label_matrix(train_origins)
validation_labels = label_matrix(validation_origins)
refit_labels = label_matrix(refit_origins)
test_labels = label_matrix(test_origins)
for origins, labels in (
    (train_origins, train_labels),
    (validation_origins, validation_labels),
    (refit_origins, refit_labels),
    (test_origins, test_labels),
):
    assert labels.shape == (len(origins), len(HORIZONS))
    assert np.isin(labels, CLASS_IDS).all()
horizon_offsets = np.asarray(HORIZONS)
assert np.all(train_origins[:, None] + horizon_offsets < validation_start)
validation_targets = validation_origins[:, None] + horizon_offsets
assert np.all((validation_targets >= validation_start) & (validation_targets < test_start))
assert np.all(refit_origins[:, None] + horizon_offsets < test_start)
assert np.all(test_origins >= test_start)
assert np.all(test_origins[:, None] + horizon_offsets < n_obs)


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


def expanding_har_ols(values: np.ndarray, origins: np.ndarray) -> np.ndarray:
    features = har_features(values)
    predictions = np.empty((len(origins), len(HORIZONS)), dtype=np.float64)
    for horizon_index, horizon in enumerate(HORIZONS):
        for position, origin in enumerate(origins):
            training = np.arange(HAR_MONTH - 1, origin - horizon + 1, dtype=int)
            x_train = features[training]
            y_train = values[training + horizon]
            finite = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)
            if finite.sum() < 60:
                raise ValueError(f"Only {finite.sum()} valid HAR rows are available at origin {origin}.")
            parameters = sm.OLS(y_train[finite], x_train[finite]).fit().params
            if not np.isfinite(parameters).all():
                raise FloatingPointError("HAR-OLS returned non-finite coefficients.")
            predictions[position, horizon_index] = float(features[origin] @ parameters)
    return predictions


test_har_level = np.exp(np.clip(expanding_har_ols(log_vix, test_origins), -20, 20))
test_har_classes = classify_vix_change(test_har_level / vix_level[test_origins, None] - 1.0)


# Statistics and classification metrics
def classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    _, recall, class_f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=CLASS_IDS,
        average=None,
        zero_division=0,
    )
    present = support > 0
    if not present.any():
        raise ValueError("At least one actual class is required.")
    return {
        "balanced_accuracy": float(recall[present].mean()),
        "macro_f1": float(class_f1[present].mean()),
        "accuracy": float(np.mean(predicted == actual)),
    }


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
    raw_statistic = differential.mean() / math.sqrt(long_run_variance / n)
    statistic = float(raw_statistic * math.sqrt(max((n - 1.0) / n, 0.0)))
    return float(2.0 * student_t.sf(abs(statistic), df=n - 1))


def bootstrap_interval(loss_differential: np.ndarray) -> tuple[float, float]:
    differential = np.asarray(loss_differential, dtype=np.float64)
    if differential.ndim != 1 or not np.isfinite(differential).all():
        raise ValueError("Bootstrap requires finite paired losses.")
    n = len(differential)
    if n < 2:
        return np.nan, np.nan
    block_length = min(max(2, int(round(n ** (1 / 3)))), n)
    starts = np.arange(n - block_length + 1)
    blocks_needed = math.ceil(n / block_length)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_REPLICATIONS)
    for replication in range(BOOTSTRAP_REPLICATIONS):
        selected_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate(
            [differential[start : start + block_length] for start in selected_starts]
        )[:n]
        means[replication] = sample.mean()
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


# Model and LoRA classifier
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = {"q", "k", "v", "o"}
LEARNING_RATE = 5e-05
TRAIN_BATCH_SIZE = 8
INFERENCE_BATCH_SIZE = 64
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


def add_lora_adapters(model: torch.nn.Module) -> None:
    model.requires_grad_(False)
    replaced_targets = set()
    for full_name, module in list(model.named_modules()):
        leaf_name = full_name.rsplit(".", 1)[-1]
        if (
            ".self_attention." not in full_name
            or leaf_name not in LORA_TARGETS
            or not isinstance(module, torch.nn.Linear)
        ):
            continue
        parent_name, _, child_name = full_name.rpartition(".")
        setattr(model.get_submodule(parent_name), child_name, LoRALinear(module))
        replaced_targets.add(leaf_name)
    if replaced_targets != LORA_TARGETS:
        raise RuntimeError(f"Incomplete Chronos-2 LoRA targets: {sorted(replaced_targets)}")


class Chronos2DirectionClassifier(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier_head = torch.nn.Linear(hidden_size, len(HORIZONS) * N_CLASSES)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        encoder_outputs, _, _, num_context_patches = self.backbone.encode(
            context=context,
            context_mask=torch.isfinite(context),
            num_output_patches=1,
        )
        pooled = encoder_outputs[0][:, num_context_patches]
        return self.classifier_head(pooled.float()).view(len(context), len(HORIZONS), N_CLASSES)


def verify_classifier_model(model: torch.nn.Module) -> None:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    has_lora = any("lora_" in name for name in trainable)
    has_head = any("classifier_head" in name for name in trainable)
    unexpected = [
        name for name in trainable if "lora_" not in name and "classifier_head" not in name
    ]
    if not has_lora or not has_head or unexpected:
        raise RuntimeError(f"Invalid trainable classifier parameters: {unexpected[:10]}")


def build_direction_classifier() -> Chronos2DirectionClassifier:
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        device_map="cuda",
        torch_dtype=torch.float32,
    )
    backbone = pipeline.model
    add_lora_adapters(backbone)
    backbone.to(DEVICE)
    model = Chronos2DirectionClassifier(backbone, int(backbone.model_dim)).to(DEVICE)
    verify_classifier_model(model)
    return model


def make_loader(contexts: np.ndarray, labels: np.ndarray, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(contexts), torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    missing = state.keys() - parameters.keys()
    if missing:
        raise RuntimeError(f"Unknown trainable-state keys: {list(missing)[:10]}")
    with torch.no_grad():
        for name, value in state.items():
            parameter = parameters[name]
            parameter.copy_(value.to(parameter.device, dtype=parameter.dtype))


def predict_probabilities(model: torch.nn.Module, contexts: np.ndarray) -> np.ndarray:
    model.eval()
    batches = []
    batch_size = min(INFERENCE_BATCH_SIZE, len(contexts))
    start = 0
    while start < len(contexts):
        stop = min(start + batch_size, len(contexts))
        context = torch.as_tensor(contexts[start:stop], dtype=torch.float32, device=DEVICE)
        try:
            with torch.inference_mode():
                logits = model(context)
            expected_shape = (stop - start, len(HORIZONS), N_CLASSES)
            if logits.shape != expected_shape or not torch.isfinite(logits).all():
                raise ValueError(f"Invalid classifier logits: {tuple(logits.shape)}")
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


def run_updates(
    model: torch.nn.Module,
    loader: DataLoader,
    maximum_updates: int,
    validation_callback: Callable[[int], bool] | None = None,
    snapshot_updates: set[int] | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    iterator = iter(loader)
    completed = accumulated = 0
    snapshots = {}
    optimizer.zero_grad(set_to_none=True)

    while completed < maximum_updates:
        try:
            contexts, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            contexts, labels = next(iterator)

        contexts = contexts.to(DEVICE)
        labels = labels.to(DEVICE)
        model.train()
        logits = model(contexts)
        expected_shape = (len(contexts), len(HORIZONS), N_CLASSES)
        if logits.shape != expected_shape:
            raise ValueError(f"Unexpected training logits: {tuple(logits.shape)}")
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
        if (
            validation_callback
            and completed % VALIDATION_EVERY_UPDATES == 0
            and validation_callback(completed)
        ):
            break

    del optimizer
    return snapshots


def select_classifier_updates(window_size: int, seed: int) -> dict[int, int]:
    set_seed(seed)
    model = build_direction_classifier()
    train_contexts = build_contexts(log_vix, train_origins, window_size)
    validation_contexts = build_contexts(log_vix, validation_origins, window_size)
    loader = make_loader(train_contexts, train_labels, seed)
    best_scores = {horizon: (-np.inf, np.inf, np.inf) for horizon in HORIZONS}
    best_updates = {horizon: -1 for horizon in HORIZONS}
    checks_without_improvement = 0

    def validate(update: int) -> bool:
        nonlocal checks_without_improvement
        probabilities = predict_probabilities(model, validation_contexts)
        improved = False
        for horizon_index, horizon in enumerate(HORIZONS):
            actual = validation_labels[:, horizon_index]
            current_probabilities = probabilities[:, horizon_index]
            predicted = current_probabilities.argmax(axis=1)
            balanced_accuracy = classification_metrics(actual, predicted)["balanced_accuracy"]
            current_log_loss = float(log_loss(actual, current_probabilities, labels=CLASS_IDS))
            candidate = (balanced_accuracy, -current_log_loss, -update)
            if update >= MIN_UPDATES and candidate > best_scores[horizon]:
                best_scores[horizon] = candidate
                best_updates[horizon] = update
                improved = True
            print(
                f"W={window_size} seed={seed} update={update} h={horizon} "
                f"balanced_accuracy={balanced_accuracy:.6f} log_loss={current_log_loss:.6f}"
            )
        checks_without_improvement = 0 if improved else checks_without_improvement + 1
        return update >= MIN_UPDATES and checks_without_improvement >= EARLY_STOPPING_PATIENCE

    run_updates(model, loader, MAX_UPDATES, validation_callback=validate)
    if any(update < 0 for update in best_updates.values()):
        raise RuntimeError(f"No classifier update selected for W={window_size}, seed={seed}.")
    del model, train_contexts, validation_contexts, loader
    gc.collect()
    torch.cuda.empty_cache()
    return best_updates


def refit_classifier(window_size: int, seed: int, selected_updates: dict[int, int]) -> np.ndarray:
    set_seed(seed)
    model = build_direction_classifier()
    refit_contexts = build_contexts(log_vix, refit_origins, window_size)
    loader = make_loader(refit_contexts, refit_labels, seed)
    required_updates = set(selected_updates.values())
    snapshots = run_updates(
        model,
        loader,
        max(required_updates),
        snapshot_updates=required_updates,
    )
    if set(snapshots) != required_updates:
        raise RuntimeError("Classifier refit did not preserve every selected update state.")

    test_contexts = build_contexts(log_vix, test_origins, window_size)
    selected_probabilities = np.empty(
        (len(test_origins), len(HORIZONS), N_CLASSES),
        dtype=np.float32,
    )
    for update in sorted(required_updates):
        restore_trainable_state(model, snapshots[update])
        probabilities = predict_probabilities(model, test_contexts)
        for horizon_index, horizon in enumerate(HORIZONS):
            if selected_updates[horizon] == update:
                selected_probabilities[:, horizon_index] = probabilities[:, horizon_index]

    del model, refit_contexts, test_contexts, loader, snapshots
    gc.collect()
    torch.cuda.empty_cache()
    return selected_probabilities


# Window loop, validation, refit and test
classifier_store: dict[int, np.ndarray] = {}
for window_size in WINDOW_SIZES:
    print(f"Training direction classifier W={window_size}")
    seed_probabilities = []
    for seed in SEEDS:
        selected_updates = select_classifier_updates(window_size, seed)
        seed_probabilities.append(refit_classifier(window_size, seed, selected_updates))
    classifier_store[window_size] = np.stack(seed_probabilities).mean(axis=0)
    np.testing.assert_allclose(classifier_store[window_size].sum(axis=-1), 1.0, atol=1e-5)
    del seed_probabilities
    gc.collect()
    torch.cuda.empty_cache()


# Result rows
result_rows = []
for window_size in WINDOW_SIZES:
    probabilities = classifier_store[window_size]
    for horizon_index, horizon in enumerate(HORIZONS):
        positions = np.arange(0, len(test_origins), horizon, dtype=int)
        actual = test_labels[positions, horizon_index]
        model_classes = probabilities[positions, horizon_index].argmax(axis=1)
        har_classes = test_har_classes[positions, horizon_index]
        model_metrics = classification_metrics(actual, model_classes)
        har_metrics = classification_metrics(actual, har_classes)
        model_loss = (model_classes != actual).astype(float)
        har_loss = (har_classes != actual).astype(float)
        differential = model_loss - har_loss
        bootstrap_lower, bootstrap_upper = bootstrap_interval(differential)
        result_rows.append(
            {
                "window_size": window_size,
                "horizon": horizon,
                "benchmark": "HAR-OLS",
                "n_obs": len(actual),
                "model_balanced_accuracy": model_metrics["balanced_accuracy"],
                "har_balanced_accuracy": har_metrics["balanced_accuracy"],
                "delta_balanced_accuracy": model_metrics["balanced_accuracy"]
                - har_metrics["balanced_accuracy"],
                "model_macro_f1": model_metrics["macro_f1"],
                "har_macro_f1": har_metrics["macro_f1"],
                "delta_macro_f1": model_metrics["macro_f1"] - har_metrics["macro_f1"],
                "model_accuracy_pct": 100.0 * model_metrics["accuracy"],
                "har_accuracy_pct": 100.0 * har_metrics["accuracy"],
                "delta_accuracy_pct_points": 100.0
                * (model_metrics["accuracy"] - har_metrics["accuracy"]),
                "dm_p_value_unadjusted": dm_p_value(differential),
                "accuracy_delta_ci_lower_pct_points": -100.0 * bootstrap_upper,
                "accuracy_delta_ci_upper_pct_points": -100.0 * bootstrap_lower,
            }
        )


# CSV
OUTPUT_COLUMNS = [
    "window_size",
    "horizon",
    "benchmark",
    "n_obs",
    "model_balanced_accuracy",
    "har_balanced_accuracy",
    "delta_balanced_accuracy",
    "model_macro_f1",
    "har_macro_f1",
    "delta_macro_f1",
    "model_accuracy_pct",
    "har_accuracy_pct",
    "delta_accuracy_pct_points",
    "dm_p_value_unadjusted",
    "accuracy_delta_ci_lower_pct_points",
    "accuracy_delta_ci_upper_pct_points",
]
results = pd.DataFrame(result_rows, columns=OUTPUT_COLUMNS).sort_values(
    ["window_size", "horizon"]
).reset_index(drop=True)
assert len(results) == len(WINDOW_SIZES) * len(HORIZONS)
assert results[["model_accuracy_pct", "har_accuracy_pct"]].apply(
    lambda column: column.between(0, 100).all()
).all()

RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
output_path = RESULTS_DIRECTORY / OUTPUT_FILENAME
results.to_csv(output_path, index=False)
print(f"Saved compact results CSV: {output_path}")


# Display
print("Primary test comparison, identical to the saved CSV")
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
    display(results)


# Plots
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for axis, horizon in zip(axes, HORIZONS):
    horizon_rows = results[results["horizon"] == horizon].sort_values("window_size")
    axis.plot(
        horizon_rows["window_size"],
        horizon_rows["model_balanced_accuracy"],
        marker="o",
        label="Chronos-2 ensemble",
    )
    axis.plot(
        horizon_rows["window_size"],
        horizon_rows["har_balanced_accuracy"],
        marker="s",
        label="HAR-OLS",
    )
    axis.set_title(f"Balanced accuracy, h={horizon}")
    axis.set_xlabel("Window")
    axis.set_ylabel("Balanced accuracy")
    axis.grid(alpha=0.25)
axes[0].legend()
fig.tight_layout()
plt.show()
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
bar_positions = np.arange(len(WINDOW_SIZES))
bar_width = 0.38
for axis, horizon in zip(axes, HORIZONS):
    horizon_rows = results[results["horizon"] == horizon].set_index("window_size").reindex(WINDOW_SIZES)
    model_bars = axis.bar(
        bar_positions - bar_width / 2,
        horizon_rows["model_accuracy_pct"],
        width=bar_width,
        label="Chronos-2 ensemble",
    )
    har_bars = axis.bar(
        bar_positions + bar_width / 2,
        horizon_rows["har_accuracy_pct"],
        width=bar_width,
        label="HAR-OLS",
    )
    axis.bar_label(model_bars, fmt="%.1f", padding=2, fontsize=8)
    axis.bar_label(har_bars, fmt="%.1f", padding=2, fontsize=8)
    axis.set_xticks(bar_positions, WINDOW_SIZES)
    axis.set_xlabel("Window")
    axis.set_title(f"Accuracy (%), h={horizon}")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()
fig.tight_layout()
plt.show()
plt.close(fig)


# Final assertions
assert set(classifier_store) == set(WINDOW_SIZES)
for probabilities in classifier_store.values():
    assert probabilities.shape == (len(test_origins), len(HORIZONS), N_CLASSES)
    assert np.isfinite(probabilities).all()
    assert np.isin(probabilities.argmax(axis=-1), CLASS_IDS).all()
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-5)
assert OUTPUT_FILENAME.count(".csv") == 1 and "_w" not in OUTPUT_FILENAME
print("All classifier protocol assertions passed.")

"""## TimesFM 2.5

### Zero Shot
"""

# Commented out IPython magic to ensure Python compatibility.

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q --upgrade-strategy only-if-needed "transformers==5.3.0"

