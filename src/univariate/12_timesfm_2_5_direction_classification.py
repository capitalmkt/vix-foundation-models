"""
12. TimesFM 2.5 - three-class direction classifier (univariate)

Model family: TimesFM 2.5
Adaptation: three-class direction classifier
Information set: VIX univariate only
Colab layout: run this complete file as one cell on a T4 GPU
Fixed windows: 32, 64, 128, 256. No window selection is performed.
Fixed horizons: 1, 5, 22 trading days.
Labels: DOWN_CLASS=0, NEUTRAL_CLASS=1, UP_CLASS=2 with a fixed 1% level-move threshold.
Model ID: google/timesfm-2.5-200m-transformers
Checkpoint revision: 5a9806b9b291fad9233b5249d88263f1846304d3
Training: five fixed seeds, one fixed family-specific LoRA configuration, unweighted cross-entropy.
Selection: validation balanced accuracy, then log loss, then fewer optimizer updates.
Ensemble: arithmetic mean of the five seed probability tensors, followed by argmax.
Benchmark: HAR-OLS using Corsi's daily/weekly/monthly structure on log-VIX with direct horizons.
Output configuration: one compact 12-row table is displayed and saved unchanged to this cell's CSV.
Plot configuration: three balanced-accuracy line panels and one grouped accuracy-percentage bar figure.
Missing-class policy: balanced accuracy and macro-F1 average only classes present in each evaluated subset.
Live data policy: disabled. An approved VIX CSV is mandatory.

Documentation:
  https://github.com/google-research/timesfm
  https://huggingface.co/google/timesfm-2.5-200m-transformers
  https://huggingface.co/google-research/timesfm-2.5-200m-transformers/commit/5a9806b9b291fad9233b5249d88263f1846304d3
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/timesfm2_5/modular_timesfm2_5.py

Installation notes:
  pip install "transformers==5.3.0" "peft==0.18.1"

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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn.functional as F
from IPython.display import display
from peft import LoraConfig, get_peft_model
from scipy.stats import t as student_t
from sklearn.metrics import log_loss, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from transformers import TimesFm2_5ModelForPrediction

# Configuration
MODEL_ID = "google/timesfm-2.5-200m-transformers"
MODEL_REVISION = "5a9806b9b291fad9233b5249d88263f1846304d3"
WINDOW_SIZES = [32, 64, 128, 256]
HORIZONS = [1, 5, 22]
MAX_WINDOW = max(WINDOW_SIZES)
MAX_HORIZON = max(HORIZONS)
HAR_WEEK = 5
HAR_MONTH = 22
SEEDS = [11, 22, 33, 44, 55]
DOWN_CLASS = 0
NEUTRAL_CLASS = 1
UP_CLASS = 2
N_CLASSES = 3
MOVE_THRESHOLD = 0.01
BOOTSTRAP_REPLICATIONS = 5000
BOOTSTRAP_SEED = 2026
OUTPUT_FILENAME = "classifier_timesfm_2_5_univariate.csv"
LOCAL_VIX_PATHS = [REPO_DATA_DIR / "VIX.csv", Path("/content/VIX.csv"), Path("/content/VIXCLS.csv"), Path("/content/vix.csv")]
RESULTS_DIRECTORY = resolve_results_directory()
enable_figure_autosave(RESULTS_DIRECTORY, Path(OUTPUT_FILENAME).stem)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type != "cuda":
    raise RuntimeError("Select a T4 GPU runtime in Google Colab before running this cell.")
print(f"GPU: {torch.cuda.get_device_name(0)}")


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
    data = data.loc[data["Date"].notna() & data["VIX"].gt(0)].sort_values("Date").reset_index(drop=True)
    if data.empty:
        raise ValueError("The approved CSV contains no valid positive VIX observations.")
    if data["Date"].duplicated().any():
        raise ValueError("Duplicate VIX dates found.")
    print(f"VIX source: {path} | n={len(data)}")
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

if min(map(len, (train_origins, validation_origins, refit_origins, test_origins))) == 0:
    raise ValueError("The cleaned VIX history is too short for the fixed 60/20/20 protocol.")


# Contexts and labels
def build_contexts(values: np.ndarray, origins: np.ndarray, window_size: int) -> np.ndarray:
    offsets = np.arange(1 - window_size, 1)
    return values[origins[:, None] + offsets].astype(np.float32, copy=False)


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


def expanding_har_ols(values: np.ndarray, origins: np.ndarray) -> np.ndarray:
    features = har_features(values)
    predictions = np.empty((len(origins), len(HORIZONS)))
    for horizon_index, horizon in enumerate(HORIZONS):
        for row_index, origin in enumerate(origins):
            training = np.arange(HAR_MONTH - 1, origin - horizon + 1)
            model = sm.OLS(values[training + horizon], features[training]).fit()
            predictions[row_index, horizon_index] = features[origin] @ model.params
    return predictions


test_har_level = np.exp(np.clip(expanding_har_ols(log_vix, test_origins), -20, 20))
test_har_classes = np.column_stack(
    [
        classify_vix_change(test_har_level[:, index] / vix_level[test_origins] - 1.0)
        for index in range(len(HORIZONS))
    ]
)


# Statistics and classification metrics
def classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    _, recall, class_f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=[DOWN_CLASS, NEUTRAL_CLASS, UP_CLASS],
        average=None,
        zero_division=0,
    )
    present = support > 0
    return (
        float(recall[present].mean()),
        float(class_f1[present].mean()),
        float(np.mean(predicted == actual)),
    )


def diebold_mariano_p_value(loss_differential: np.ndarray) -> float:
    differential = np.asarray(loss_differential, dtype=np.float64)
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


def moving_block_bootstrap(loss_differential: np.ndarray) -> tuple[float, float]:
    differential = np.asarray(loss_differential, dtype=np.float64)
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
LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
LEARNING_RATE = 0.0001
TRAIN_BATCH_SIZE = 16
INFERENCE_BATCH_SIZE = 64
MIN_UPDATES = 100
VALIDATION_EVERY_UPDATES = 50
MAX_UPDATES = 500
EARLY_STOPPING_PATIENCE = 3
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0


class TimesFmDirectionClassifier(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier_head = torch.nn.Linear(hidden_size, len(HORIZONS) * N_CLASSES)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        output = self.backbone(
            past_values=[row for row in context],
            forecast_context_len=context.shape[1],
            truncate_negative=False,
        )
        hidden = output.last_hidden_state
        if hidden.ndim != 3:
            raise ValueError(f"Unexpected backbone hidden-state shape: {tuple(hidden.shape)}")
        return self.classifier_head(hidden[:, -1].float()).view(len(context), len(HORIZONS), N_CLASSES)


def build_direction_classifier() -> TimesFmDirectionClassifier:
    base_model = TimesFm2_5ModelForPrediction.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.float32,
    ).to(DEVICE)
    backbone = get_peft_model(
        base_model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules="all-linear",
            bias="none",
        ),
    ).to(DEVICE)
    hidden_size = int(backbone.get_base_model().config.hidden_size)
    return TimesFmDirectionClassifier(backbone, hidden_size).to(DEVICE)


def make_loader(contexts: np.ndarray, labels: np.ndarray, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(contexts), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)


def classifier_probabilities(model: torch.nn.Module, contexts: np.ndarray) -> np.ndarray:
    model.eval()
    batches = []
    start = 0
    batch_size = min(INFERENCE_BATCH_SIZE, len(contexts))
    while start < len(contexts):
        stop = min(start + batch_size, len(contexts))
        context = torch.as_tensor(contexts[start:stop], dtype=torch.float32, device=DEVICE)
        try:
            with torch.inference_mode():
                logits = model(context)
            if logits.shape != (stop - start, len(HORIZONS), N_CLASSES):
                raise ValueError(f"Unexpected classifier logits shape: {tuple(logits.shape)}")
            batches.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
            start = stop
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            warnings.warn(f"CUDA OOM: inference batch reduced to {batch_size}.", RuntimeWarning)
    return np.concatenate(batches)


def run_updates(
    model: torch.nn.Module,
    loader: DataLoader,
    maximum_updates: int,
    validation_callback=None,
    checkpoint_updates: set[int] | None = None,
) -> tuple[dict[int, dict[str, torch.Tensor]], int]:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    checkpoint_updates = checkpoint_updates or set()
    snapshots = {}
    iterator = iter(loader)
    completed = 0

    while completed < maximum_updates:
        try:
            contexts, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            contexts, labels = next(iterator)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(contexts.to(DEVICE))
        loss = F.cross_entropy(logits.reshape(-1, N_CLASSES), labels.to(DEVICE).reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError("Classifier loss is not finite.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        completed += 1

        if completed in checkpoint_updates:
            snapshots[completed] = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        if (
            validation_callback is not None
            and completed % VALIDATION_EVERY_UPDATES == 0
            and validation_callback(completed)
        ):
            break

    del optimizer
    return snapshots, completed


def select_classifier_updates(
    train_contexts: np.ndarray,
    validation_contexts: np.ndarray,
    seed: int,
) -> dict[int, int]:
    set_seed(seed)
    model = build_direction_classifier()
    loader = make_loader(train_contexts, train_labels, seed)
    best_scores = {horizon: (-np.inf, np.inf, np.inf) for horizon in HORIZONS}
    best_updates = {horizon: -1 for horizon in HORIZONS}
    checks_without_improvement = 0

    def validate(update: int) -> bool:
        nonlocal checks_without_improvement
        probabilities = classifier_probabilities(model, validation_contexts)
        improved = False
        for horizon_index, horizon in enumerate(HORIZONS):
            actual = validation_labels[:, horizon_index]
            horizon_probabilities = probabilities[:, horizon_index]
            balanced_accuracy, _, _ = classification_metrics(actual, horizon_probabilities.argmax(axis=1))
            current_log_loss = float(
                log_loss(actual, horizon_probabilities, labels=[DOWN_CLASS, NEUTRAL_CLASS, UP_CLASS])
            )
            candidate = (balanced_accuracy, -current_log_loss, -update)
            if update >= MIN_UPDATES and candidate > best_scores[horizon]:
                best_scores[horizon] = candidate
                best_updates[horizon] = update
                improved = True
        checks_without_improvement = 0 if improved else checks_without_improvement + 1
        return update >= MIN_UPDATES and checks_without_improvement >= EARLY_STOPPING_PATIENCE

    run_updates(model, loader, MAX_UPDATES, validation_callback=validate)
    if any(update < 0 for update in best_updates.values()):
        raise RuntimeError("Validation did not select an optimizer update for every horizon.")

    del model, loader
    gc.collect()
    torch.cuda.empty_cache()
    return best_updates


def refit_classifier(
    refit_contexts: np.ndarray,
    test_contexts: np.ndarray,
    seed: int,
    selected_updates: dict[int, int],
) -> np.ndarray:
    set_seed(seed)
    model = build_direction_classifier()
    loader = make_loader(refit_contexts, refit_labels, seed)
    required_updates = set(selected_updates.values())
    snapshots, completed = run_updates(
        model,
        loader,
        max(required_updates),
        checkpoint_updates=required_updates,
    )
    if completed != max(required_updates) or set(snapshots) != required_updates:
        raise RuntimeError("Classifier refit did not preserve every selected update state.")

    parameters = dict(model.named_parameters())
    horizon_probabilities = []
    for horizon_index, horizon in enumerate(HORIZONS):
        with torch.no_grad():
            for name, value in snapshots[selected_updates[horizon]].items():
                parameters[name].copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))
        probabilities = classifier_probabilities(model, test_contexts)
        horizon_probabilities.append(probabilities[:, horizon_index])

    result = np.stack(horizon_probabilities, axis=1)
    del model, loader, snapshots
    gc.collect()
    torch.cuda.empty_cache()
    return result


# Window loop, validation, refit and test
classifier_probabilities_by_window = {}
for window_size in WINDOW_SIZES:
    print(f"Training direction classifier W={window_size}")
    train_contexts = build_contexts(log_vix, train_origins, window_size)
    validation_contexts = build_contexts(log_vix, validation_origins, window_size)
    refit_contexts = build_contexts(log_vix, refit_origins, window_size)
    test_contexts = build_contexts(log_vix, test_origins, window_size)
    seed_probabilities = []

    for seed in SEEDS:
        selected_updates = select_classifier_updates(train_contexts, validation_contexts, seed)
        print(f"W={window_size} seed={seed} selected updates: {selected_updates}")
        seed_probabilities.append(
            refit_classifier(refit_contexts, test_contexts, seed, selected_updates)
        )

    classifier_probabilities_by_window[window_size] = np.stack(seed_probabilities).mean(axis=0)
    del train_contexts, validation_contexts, refit_contexts, test_contexts, seed_probabilities
    gc.collect()


# Result rows
result_rows = []
for window_size in WINDOW_SIZES:
    probabilities = classifier_probabilities_by_window[window_size]
    for horizon_index, horizon in enumerate(HORIZONS):
        positions = np.arange(0, len(test_origins), horizon)
        actual = test_labels[positions, horizon_index]
        model_probabilities = probabilities[positions, horizon_index]
        model_classes = model_probabilities.argmax(axis=1)
        har_classes = test_har_classes[positions, horizon_index]

        model_balanced, model_f1, model_accuracy = classification_metrics(actual, model_classes)
        har_balanced, har_f1, har_accuracy = classification_metrics(actual, har_classes)
        model_loss = (model_classes != actual).astype(float)
        har_loss = (har_classes != actual).astype(float)
        loss_differential = model_loss - har_loss
        bootstrap_lower, bootstrap_upper = moving_block_bootstrap(loss_differential)

        result_rows.append(
            {
                "window_size": window_size,
                "horizon": horizon,
                "benchmark": "HAR-OLS",
                "n_obs": len(actual),
                "model_balanced_accuracy": model_balanced,
                "har_balanced_accuracy": har_balanced,
                "delta_balanced_accuracy": model_balanced - har_balanced,
                "model_macro_f1": model_f1,
                "har_macro_f1": har_f1,
                "delta_macro_f1": model_f1 - har_f1,
                "model_accuracy_pct": 100.0 * model_accuracy,
                "har_accuracy_pct": 100.0 * har_accuracy,
                "delta_accuracy_pct_points": 100.0 * (model_accuracy - har_accuracy),
                "dm_p_value_unadjusted": diebold_mariano_p_value(loss_differential),
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
    axis.plot(horizon_rows["window_size"], horizon_rows["model_balanced_accuracy"], marker="o", label="TimesFM 2.5 ensemble")
    axis.plot(horizon_rows["window_size"], horizon_rows["har_balanced_accuracy"], marker="s", label="HAR-OLS")
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
        label="TimesFM 2.5 ensemble",
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
assert results.columns.tolist() == OUTPUT_COLUMNS
assert len(results) == len(WINDOW_SIZES) * len(HORIZONS)
assert results[["model_accuracy_pct", "har_accuracy_pct"]].apply(
    lambda column: column.between(0, 100).all()
).all()
print("All classifier protocol assertions passed.")

"""# Multivariate Models

## Chronos Bolt

### Zero Shot
"""

# Commented out IPython magic to ensure Python compatibility.

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q --upgrade-strategy only-if-needed "chronos-forecasting==2.3.1"

