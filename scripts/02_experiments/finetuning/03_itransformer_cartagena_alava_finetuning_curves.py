"""
Cross-greenhouse FINE-TUNING experiments (Cartagena -> Alava) with iTransformer.
SCRIPT 1 of 2: convergence curve (epochs) and learning curve (fine-tuning days).

This script reuses the architecture and preprocessing from
`scripts/02_experiments/alava/10_itransformer_alava.py`.

Phases:

  PHASE 0 (pre-training, once only):
    Trains the model from scratch with the initial 70% of Cartagena
    (interleaved train/val blocks, same as the base script). The
    checkpoint is saved to disk so this phase doesn't have to be repeated
    for every experiment.

  PHASE 1 (Experiment A - Convergence curve):
    Starts from the Cartagena checkpoint and fine-tunes with 100% of the
    initial 70% of Alava, WITHOUT early stopping, logging
    train_loss/val_loss epoch by epoch. Used to decide how many epochs the
    fine-tuning needs to converge. Also evaluates on Alava's fixed
    test set every few epochs to check that the val_loss minimum matches
    the best actual test performance.

  PHASE 2 (Experiment B - Learning curve by amount of data):
    Repeats the fine-tuning (always starting from the same Cartagena
    checkpoint) using increasing amounts of Alava days for train (the
    val and test of Alava stay FIXED in every case). Each size is
    repeated with several seeds. For each data size, it saves: the final
    test metric and the number of epochs needed until early stopping.

Alava's final 30% is NEVER used for training or fine-tuning: it is
the only test set, fixed, for both phases.
"""

import copy
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm


# =============================================================================
# General utilities
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Column name normalization between datasets from different sites
# =============================================================================
SITE_SUFFIXES = ["_Alava", "_Cartagena_comp0", "_Cartagena_meteo"]


def normalize_columns(df, suffixes=SITE_SUFFIXES):
    rename_map = {}
    for col in df.columns:
        new_name = col
        for suffix in suffixes:
            if new_name.endswith(suffix):
                new_name = new_name[: -len(suffix)]
                break
        rename_map[col] = new_name

    df = df.rename(columns=rename_map)

    duplicates = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(
            f"After normalizing columns, duplicate names remain: {duplicates}. "
            "Check the site suffixes or the source CSV."
        )
    return df


# =============================================================================
# Dataset / sequences / blocks
# =============================================================================
class ITransformerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_multivariate_sequences(df, variate_columns, window_size, horizon, stride=1):
    X, y = [], []
    data = df[variate_columns].values

    limite = len(df) - window_size - horizon + 1
    for i in range(0, max(limite, 0), stride):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + horizon])

    return np.array(X), np.array(y)


def label_day_blocks(df, train_days=3, val_days=1):
    normalized_days = pd.Series(df.index.normalize(), index=df.index)
    unique_days = sorted(normalized_days.unique())

    cycle = train_days + val_days
    label_by_day = {
        day: ("train" if (i % cycle) < train_days else "val")
        for i, day in enumerate(unique_days)
    }

    labels = normalized_days.map(label_by_day)
    block_id = (labels != labels.shift()).cumsum()

    return labels, block_id


def split_into_blocks(df, labels, block_id):
    blocks = []
    for bid, idx in block_id.groupby(block_id).groups.items():
        sub_df = df.loc[idx]
        label = labels.loc[idx].iloc[0]
        blocks.append((label, sub_df))

    blocks.sort(key=lambda b: b[1].index[0])
    return blocks


def generate_sequences_by_blocks(blocks, variate_columns, window_size, horizon, stride):
    Xs, ys = [], []
    for b in blocks:
        if len(b) < window_size + horizon:
            continue
        xb, yb = prepare_multivariate_sequences(
            b, variate_columns, window_size, horizon, stride=stride
        )
        if len(xb) > 0:
            Xs.append(xb)
            ys.append(yb)

    if not Xs:
        raise ValueError("Could not generate any sequence from these blocks.")

    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def select_first_days(train_blocks, target_days, samples_per_day):
    """
    From a list of TRAIN blocks already sorted chronologically, returns a
    new list of blocks containing, in total, approximately `target_days`
    days of data (the first ones chronologically), truncating the last
    block if needed so as not to overshoot.

    If `target_days` is None, returns all blocks as-is (100%).
    """
    if target_days is None:
        return train_blocks

    target_rows = int(round(target_days * samples_per_day))
    if target_rows <= 0:
        return []

    selection = []
    accumulated_rows = 0
    for b in train_blocks:
        if accumulated_rows >= target_rows:
            break
        remaining_rows = target_rows - accumulated_rows
        if len(b) <= remaining_rows:
            selection.append(b)
            accumulated_rows += len(b)
        else:
            selection.append(b.iloc[:remaining_rows])
            accumulated_rows += remaining_rows
    return selection


# =============================================================================
# iTransformer model
# =============================================================================
class InvertedEmbedding(nn.Module):
    def __init__(self, window_size, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(window_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.dropout(self.value_embedding(x))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x):
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        y = self.dropout(self.activation(self.linear1(x)))
        y = self.dropout(self.linear2(y))
        x = self.norm2(x + y)

        return x, attn_weights


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x):
        attns = []
        for layer in self.layers:
            x, attn = layer(x)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class Model(nn.Module):
    def __init__(self, n_vars, window_size, horizon,
                 d_model=128, n_heads=8, e_layers=2, d_ff=128,
                 dropout=0.1, activation="gelu", use_norm=True):
        super().__init__()
        self.use_norm = use_norm

        self.embedding = InvertedEmbedding(window_size, d_model, dropout)
        self.encoder = Encoder(
            [EncoderLayer(d_model, n_heads, d_ff, dropout, activation) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, horizon)

    def forward(self, x):
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x = x - means
            stds = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / stds

        tokens = self.embedding(x)
        enc_out, _ = self.encoder(tokens)
        dec_out = self.projector(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stds.repeat(1, dec_out.size(1), 1) + means.repeat(1, dec_out.size(1), 1)

        return dec_out


def create_model(n_vars, window_size, horizon, cfg, device):
    return Model(
        n_vars=n_vars,
        window_size=window_size,
        horizon=horizon,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        e_layers=cfg["e_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        use_norm=cfg["use_norm"],
    ).to(device)


# =============================================================================
# Metrics
# =============================================================================
def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100),
        "relative_error": float(np.mean((y_pred - y_true) / (y_true + 1e-8))),
        "relative_error_abs": float(np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8)))),
    }


def flatten_metrics(metrics_por_target):
    """
    {"DB_TEMP_INT": {"r2":..., "rmse":...}, "REL_HUM_INT": {...}}
    ->
    {"DB_TEMP_INT_r2":..., "DB_TEMP_INT_rmse":..., "REL_HUM_INT_r2":..., ...}
    """
    flat = {}
    for target_label, m in metrics_por_target.items():
        if m is None:
            continue
        for k, v in m.items():
            flat[f"{target_label}_{k}"] = v
    return flat


def evaluate_on_test(model, df_test, variate_columns, target_idx, window_size, horizon, scaler_X, device, step=None):
    """
    Lightweight evaluation: sweeps the test set with a sliding window,
    predicts all variates and returns only the global metrics per target.
    Metrics are computed in actual units (inverse MinMax scaling).
    """
    model.eval()
    step = step or horizon

    rows_by_target = {name: {"actual": [], "pred": []} for name in target_idx}

    with torch.no_grad():
        for i in range(0, len(df_test) - window_size - horizon + 1, step):
            x_in = df_test[variate_columns].iloc[i:i + window_size].values
            x_in_t = torch.tensor(x_in, dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(x_in_t).cpu().numpy()[0]  # (horizon, n_vars), scaled

            actual_block = df_test[variate_columns].iloc[
                i + window_size: i + window_size + horizon
            ].values  # (horizon, n_vars), scaled

            pred_desc = scaler_X.inverse_transform(pred)
            actual_desc = scaler_X.inverse_transform(actual_block)

            for name, idx in target_idx.items():
                rows_by_target[name]["actual"].extend(actual_desc[:, idx].tolist())
                rows_by_target[name]["pred"].extend(pred_desc[:, idx].tolist())

    metrics = {}
    for name, d in rows_by_target.items():
        if len(d["actual"]) < 2:
            metrics[name] = None
            continue
        metrics[name] = calculate_metrics(d["actual"], d["pred"])
    return metrics


# =============================================================================
# Reusable training
# =============================================================================
def run_train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def run_eval_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total += loss.item()
    return total / len(loader)


def train(
    model, train_loader, val_loader, device, lr,
    num_epochs, patience, use_scheduler=True,
    mostrar_barra=False, log_prefix="",
):
    """
    Generic training loop (pre-training on Cartagena and fine-tuning on Alava).

    Returns:
      history, best_state, best_epoch, epochs_run, early_stopped
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    counter = 0
    early_stopped = False

    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        loader = tqdm(train_loader, desc=f"{log_prefix}Epoch {epoch+1} [TRAIN]", leave=False) if mostrar_barra else train_loader
        train_loss = run_train_epoch(model, loader, optimizer, criterion, device)
        val_loss = run_eval_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_loss)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"{log_prefix}Epoch {epoch+1}: train={train_loss:.5f} | val={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                early_stopped = True
                print(f"{log_prefix}Early stopping triggered at epoch {epoch+1} "
                      f"(best val_loss at epoch {best_epoch})")
                break

    epochs_run = history["epoch"][-1] if history["epoch"] else 0
    return history, best_state, best_epoch, epochs_run, early_stopped


# =============================================================================
# CONFIG
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg_modelo = {
    "d_model": 128,
    "n_heads": 8,
    "e_layers": 3,
    "d_ff": 128,
    "dropout": 0.1,
    "use_norm": False,
}

batch_size = 128
window_size = 96
horizon = 24
samples_per_day = 96  # data every 15 min -> 96 samples/day

# Pre-training on Cartagena
lr_pretrain = 1e-4
num_epochs_pretrain = 100
patience_pretrain = 10
seed_pretrain = 42

# Interleaved train/val blocks
train_block_days = 14
val_block_days = 7
stride_train = 1
stride_val = 1

# Fine-tuning
lr_finetune = 1e-4  # if fine-tuning is unstable, try a smaller value (e.g. 2e-5)

# ---------------------------
# Input data
# ---------------------------
# csv1_path = source for PRE-TRAINING.
# csv2_path = FINE-TUNING target (70% fine-tuning, final 30% fixed test).
csv1_path = "./data/processed/Cartagena_comp0_processed_aligned.csv"
csv2_path = "./data/processed/Alava_processed_aligned.csv"

csv1_name = os.path.splitext(os.path.basename(csv1_path))[0]
csv2_name = os.path.splitext(os.path.basename(csv2_path))[0]

variate_columns = [
    'DB_TEMP_INT_INPUT',
    'DB_TEMP_EXT',
    'REL_HUM_EXT',
    'RADIATION_MEAN',
    'REL_HUM_INT',
    'month_sin',
    'month_cos',
    'is_day',
]

target_columns = {
    "DB_TEMP_INT": "DB_TEMP_INT",
    "REL_HUM_INT": "REL_HUM_INT",
}

for col in target_columns.values():
    if col not in variate_columns:
        variate_columns.append(col)

target_idx = {name: variate_columns.index(col) for name, col in target_columns.items()}

# ---------------------------
# Output paths: results/finetuning/<source>_to_<destination>/

# ---------------------------
SOURCE_GREENHOUSE = "cartagena"
DESTINATION_GREENHOUSE = "alava"

RESULTS_DIR = f"./results/finetuning/{SOURCE_GREENHOUSE}_to_{DESTINATION_GREENHOUSE}"
CHECKPOINTS_DIR = f"{RESULTS_DIR}/checkpoints"
CSV_DIR = f"{RESULTS_DIR}/csv"

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


# =============================================================================
# Loading and column normalization for both CSVs
# =============================================================================
def load_datasets():
    df1 = pd.read_csv(csv1_path, index_col=0, parse_dates=True)
    df2 = pd.read_csv(csv2_path, index_col=0, parse_dates=True)

    df1 = normalize_columns(df1)
    df2 = normalize_columns(df2)

    for df_name, check_df, path in [
        (csv1_name, df1, csv1_path),
        (csv2_name, df2, csv2_path),
    ]:
        missing_columns = [c for c in variate_columns if c not in check_df.columns]
        if missing_columns:
            raise ValueError(
                f"Dataset '{df_name}' ({path}) is missing these columns after "
                f"normalizing: {missing_columns}"
            )
    return df1, df2


# =============================================================================
# PHASE 0: pre-training on Cartagena (once only, cached on disk)
# =============================================================================
def prepare_and_pretrain(df1, df2, force_retrain=False):
    # --- 70/30 split on each CSV ---
    split_70_csv1 = int(len(df1) * 0.7)
    df1_trainval = df1.iloc[:split_70_csv1].copy()

    split_70_csv2 = int(len(df2) * 0.7)
    df2_trainval = df2.iloc[:split_70_csv2].copy()
    df2_test = df2.iloc[split_70_csv2:].copy()

    # --- Cartagena train/val blocks (pre-training) ---
    source_labels, source_block_id = label_day_blocks(df1_trainval, train_block_days, val_block_days)
    source_blocks = split_into_blocks(df1_trainval, source_labels, source_block_id)
    source_train_blocks = [sub for label, sub in source_blocks if label == "train"]
    source_val_blocks = [sub for label, sub in source_blocks if label == "val"]

    print(f"[{csv1_name}] train blocks: {len(source_train_blocks)} | val: {len(source_val_blocks)}")

    # --- Alava train/val blocks (fine-tuning) ---
    destination_labels, destination_block_id = label_day_blocks(df2_trainval, train_block_days, val_block_days)
    destination_blocks = split_into_blocks(df2_trainval, destination_labels, destination_block_id)
    destination_train_blocks = [sub for label, sub in destination_blocks if label == "train"]
    destination_val_blocks = [sub for label, sub in destination_blocks if label == "val"]

    print(f"[{csv2_name}] train blocks (fine-tuning): {len(destination_train_blocks)} | "
          f"val (fixed): {len(destination_val_blocks)} | test (fixed, 30%): {len(df2_test)} rows")

    available_destination_days = sum(len(b) for b in destination_train_blocks) / samples_per_day
    print(f"[{csv2_name}] available fine-tuning train days: ~{available_destination_days:.1f}")

    # --- scaler fit ONLY on Cartagena's train ---
    scaler_X = MinMaxScaler()
    concatenated_source_train_df = pd.concat(source_train_blocks)
    scaler_X.fit(concatenated_source_train_df[variate_columns])

    def scale(sub_df):
        sub_df = sub_df.copy()
        sub_df[variate_columns] = scaler_X.transform(sub_df[variate_columns])
        return sub_df

    scaled_source_train_blocks = [scale(b) for b in source_train_blocks]
    scaled_source_val_blocks = [scale(b) for b in source_val_blocks]
    scaled_destination_train_blocks = [scale(b) for b in destination_train_blocks]
    scaled_destination_val_blocks = [scale(b) for b in destination_val_blocks]

    scaled_destination_test_df = df2_test.copy()
    scaled_destination_test_df[variate_columns] = scaler_X.transform(scaled_destination_test_df[variate_columns])

    n_vars = len(variate_columns)
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"itransformer_pretrained_{csv1_name}.pt")

    if os.path.exists(checkpoint_path) and not force_retrain:
        print(f"Loading existing pretrained checkpoint: {checkpoint_path}")
        pretrained_state = torch.load(checkpoint_path, map_location=device)
    else:
        print(f"Pre-training on {csv1_name} (this only happens once)...")
        source_X_train, source_y_train = generate_sequences_by_blocks(
            scaled_source_train_blocks, variate_columns, window_size, horizon, stride=stride_train
        )
        source_X_val, source_y_val = generate_sequences_by_blocks(
            scaled_source_val_blocks, variate_columns, window_size, horizon, stride=stride_val
        )
        source_train_loader = DataLoader(ITransformerDataset(source_X_train, source_y_train), batch_size=batch_size, shuffle=True)
        source_val_loader = DataLoader(ITransformerDataset(source_X_val, source_y_val), batch_size=batch_size, shuffle=False)

        set_seed(seed_pretrain)
        pretrain_model = create_model(n_vars, window_size, horizon, cfg_modelo, device)

        inicio = time.time()
        _, pretrained_state, best_epoch, _, _ = train(
            pretrain_model, source_train_loader, source_val_loader, device,
            lr=lr_pretrain, num_epochs=num_epochs_pretrain, patience=patience_pretrain,
            mostrar_barra=True, log_prefix=f"[Pretrain-{csv1_name}] ",
        )
        print(f"Pre-training finished in {time.time() - inicio:.1f}s (best epoch: {best_epoch})")
        torch.save(pretrained_state, checkpoint_path)

    context = {
        "n_vars": n_vars,
        "scaled_destination_train_blocks": scaled_destination_train_blocks,
        "scaled_destination_val_blocks": scaled_destination_val_blocks,
        "scaled_destination_test_df": scaled_destination_test_df,
        "pretrained_state": pretrained_state,
        "available_destination_days": available_destination_days,
        "scaler_X": scaler_X,
    }
    return context


# =============================================================================
# EXPERIMENT A: fine-tuning convergence curve
# =============================================================================
def _run_convergence_single_seed(context, num_epochs, seed, eval_test_cada, patience_ref=10):
    destination_val_loader = DataLoader(
        ITransformerDataset(*generate_sequences_by_blocks(
            context["scaled_destination_val_blocks"], variate_columns, window_size, horizon, stride=stride_val
        )),
        batch_size=batch_size, shuffle=False,
    )

    destination_X_train, destination_y_train = generate_sequences_by_blocks(
        context["scaled_destination_train_blocks"], variate_columns, window_size, horizon, stride=stride_train
    )
    destination_train_loader = DataLoader(ITransformerDataset(destination_X_train, destination_y_train), batch_size=batch_size, shuffle=True)

    set_seed(seed)
    model = create_model(context["n_vars"], window_size, horizon, cfg_modelo, device)
    model.load_state_dict(context["pretrained_state"])

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr_finetune)

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    test_metrics_by_epoch = {}

    for epoch in range(num_epochs):
        train_loss = run_train_epoch(model, destination_train_loader, optimizer, criterion, device)
        val_loss = run_eval_epoch(model, destination_val_loader, criterion, device)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"[Convergence seed={seed}] Epoch {epoch+1}: train={train_loss:.5f} | val={val_loss:.5f}")

        if (epoch + 1) % eval_test_cada == 0 or epoch == num_epochs - 1:
            metrics_test = evaluate_on_test(
                model, context["scaled_destination_test_df"], variate_columns, target_idx, window_size, horizon,
                context["scaler_X"], device,
            )
            test_metrics_by_epoch[epoch + 1] = metrics_test

    # Epoch where a standard early-stopping would have cut off.
    best_val = float("inf")
    convergence_epoch = history["epoch"][-1]
    counter = 0
    for e, vl in zip(history["epoch"], history["val_loss"]):
        if vl < best_val:
            best_val = vl
            convergence_epoch = e
            counter = 0
        else:
            counter += 1
            if counter >= patience_ref:
                break

    return history, test_metrics_by_epoch, convergence_epoch


def convergence_experiment(
    context,
    num_epochs=150,
    n_seeds=1,              # bump this (e.g. 5) to show the variation band in the paper
    eval_test_cada=5,
    patience_ref=10,
):
    print(f"\n=== EXPERIMENT A: fine-tuning convergence curve (Cartagena -> Alava), n_seeds={n_seeds} ===")

    history_rows = []
    test_rows = []
    convergence_epochs = []

    for seed in range(n_seeds):
        history, test_metrics_by_epoch, convergence_epoch = _run_convergence_single_seed(
            context, num_epochs=num_epochs, seed=seed, eval_test_cada=eval_test_cada, patience_ref=patience_ref
        )
        convergence_epochs.append(convergence_epoch)

        for e, tl, vl in zip(history["epoch"], history["train_loss"], history["val_loss"]):
            history_rows.append({
                "seed": seed, "epoch": e, "train_loss": tl, "val_loss": vl,
                "estimated_convergence_epoch": convergence_epoch,
            })

        for epoch, metrics_por_target in test_metrics_by_epoch.items():
            test_rows.append({"seed": seed, "epoch": epoch, **flatten_metrics(metrics_por_target)})

    df_history = pd.DataFrame(history_rows)
    df_history.to_csv(os.path.join(CSV_DIR, "experiment_A_history.csv"), index=False)

    df_test_epoch = pd.DataFrame(test_rows)
    df_test_epoch.to_csv(os.path.join(CSV_DIR, "experiment_A_test_by_epoch.csv"), index=False)

    convergence_epochs = np.array(convergence_epochs)
    print(f"\n>>> Estimated convergence epoch per seed: {convergence_epochs.tolist()}")
    print(f">>> Mean: {convergence_epochs.mean():.1f} | Std: {convergence_epochs.std():.1f}")
    print("CSVs saved: experiment_A_history.csv, experiment_A_test_by_epoch.csv")

    return df_history, df_test_epoch


# =============================================================================
# EXPERIMENT B: learning curve by amount of fine-tuning days
# =============================================================================
def calculate_target_days_by_steps(available_days, step_days=7, include_total=True):
    """
    step_days, 2*step_days, ... up to `available_days` (plus the total as a
    final point if it doesn't land exactly on a step).
    """
    available_days = int(available_days)
    target_days = list(range(step_days, available_days + 1, step_days))
    if not target_days:
        target_days = [available_days]
    if include_total and target_days[-1] != available_days:
        target_days.append(available_days)
    return target_days


def learning_curve_experiment(
    context,
    step_days=7,
    n_seeds=3,
    num_epochs=100,
    patience=10,
):
    print("\n=== EXPERIMENT B: learning curve (available fine-tuning days) ===")

    available_days = context["available_destination_days"]
    target_days = calculate_target_days_by_steps(available_days, step_days=step_days)
    print(f"Target days to evaluate (step={step_days}, of ~{available_days:.1f} available): {target_days}")

    destination_val_loader = DataLoader(
        ITransformerDataset(*generate_sequences_by_blocks(
            context["scaled_destination_val_blocks"], variate_columns, window_size, horizon, stride=stride_val
        )),
        batch_size=batch_size, shuffle=False,
    )

    results = []

    for days in target_days:
        sub_blocks = select_first_days(context["scaled_destination_train_blocks"], days, samples_per_day)
        try:
            X_sub, y_sub = generate_sequences_by_blocks(
                sub_blocks, variate_columns, window_size, horizon, stride=stride_train
            )
        except ValueError:
            print(f"[days={days}] not enough data to generate sequences, skipping.")
            continue

        for seed in range(n_seeds):
            print(f"\n--- days={days} | n_sequences={len(y_sub)} | seed={seed} ---")
            set_seed(seed)

            train_loader = DataLoader(ITransformerDataset(X_sub, y_sub), batch_size=batch_size, shuffle=True)

            model = create_model(context["n_vars"], window_size, horizon, cfg_modelo, device)
            model.load_state_dict(context["pretrained_state"])

            _, best_state, best_epoch, epochs_run, early_stopped = train(
                model, train_loader, destination_val_loader, device,
                lr=lr_finetune, num_epochs=num_epochs, patience=patience,
                log_prefix=f"[days={days} seed={seed}] ",
            )
            model.load_state_dict(best_state)

            metrics_test = evaluate_on_test(
                model, context["scaled_destination_test_df"], variate_columns, target_idx, window_size, horizon,
                context["scaler_X"], device,
            )

            results.append({
                "finetune_days": days,
                "n_train_sequences": len(y_sub),
                "seed": seed,
                "best_val_epoch": best_epoch,
                "epochs_run": epochs_run,
                "early_stopped": early_stopped,
                **flatten_metrics(metrics_test),
            })

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("No results in experiment B (check target_days / available data).")
        return df_res

    df_res = df_res.sort_values(["finetune_days", "seed"])
    df_res.to_csv(os.path.join(CSV_DIR, "experiment_B_raw.csv"), index=False)

    metric_columns = [c for c in df_res.columns if c not in (
        "finetune_days", "n_train_sequences", "seed", "epochs_run", "early_stopped"
    )]
    agg_dict = {"n_train_sequences": ("n_train_sequences", "mean")}
    for col in metric_columns:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"] = (col, "std")

    aggregated = df_res.groupby("finetune_days").agg(**agg_dict).reset_index().sort_values("finetune_days")
    aggregated.to_csv(os.path.join(CSV_DIR, "experiment_B_aggregated.csv"), index=False)

    print("CSVs saved: experiment_B_raw.csv (one row per seed), "
          "experiment_B_aggregated.csv (mean/std per day size)")

    return df_res


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    df1, df2 = load_datasets()
    context = prepare_and_pretrain(df1, df2)

    convergence_experiment(context, num_epochs=150)
    learning_curve_experiment(context, step_days=7, n_seeds=3, num_epochs=100, patience=10)

    print(f"\nDone. Experiment A and B results are under '{CSV_DIR}' as CSV, "
          f"ready to load into a notebook.")
