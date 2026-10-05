"""
Cross-greenhouse transfer (Alava -> Cartagena) with iTransformer.
SCRIPT 2 of 2: direct comparison, zero-shot transfer vs fine-tuning vs same greenhouse.

This script reuses the architecture and preprocessing from
`scripts/02_experiments/alava/10_itransformer_alava.py`.

Phases:

  PHASE 0 (pre-training):
    Trains the model from scratch with the initial 70% of Alava
    (interleaved train/val blocks, same as the base script).

  EXPERIMENT C (direct comparison):
    1) Zero-shot: the model just trained on Alava, as-is, evaluated on
       Cartagena's fixed test.
    2) Fine-tuned: the same pre-trained weights fine-tuned with Cartagena
       days (all available by default), evaluated on the same fixed test.
    3) Same greenhouse (reference): a model trained from scratch only on
       Cartagena's train/val blocks, with the same input variables and its
       own scaler, evaluated on the same fixed test.
    Full evaluation (global + per_horizon + per_month + day/night), saved as
    JSON with the same format as 10_itransformer_alava.py so it can be
    loaded with the same analysis notebook. The JSON files are named after
    this script.

Cartagena's final 30% is NEVER used for training or fine-tuning.
"""

import copy
import json
import os
import time
import uuid

import ephem
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
# Column name normalization / day-night
# =============================================================================
SITE_SUFFIXES = ["_Alava", "_Cartagena_comp0", "_Cartagena_meteo"]

# Same coordinates as the base script (10_itransformer_alava.py), so
# day/night is computed the same way and metrics stay comparable.
LAT = 42.853615
LON = -2.674332


def is_it_day_from_date(date, lat, lon):
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.date = date
    sun = ephem.Sun(observer)
    return observer.previous_rising(sun) > observer.previous_setting(sun)


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
    Returns the first ~`target_days` days of the (chronologically sorted)
    TRAIN blocks, truncating the last block if needed.
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
    Generic training loop (pre-training on Alava and fine-tuning on Cartagena).

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

# Pre-training on Alava
lr_pretrain = 1e-4
num_epochs_pretrain = 100
patience_pretrain = 10

# Interleaved train/val blocks
train_block_days = 14
val_block_days = 7
stride_train = 1
stride_val = 1

# Fine-tuning
lr_finetune = 1e-4

# ---------------------------
# Input data
# ---------------------------
csv1_path = "./data/processed/Alava_processed_aligned.csv"
csv2_path = "./data/processed/Cartagena_comp0_processed_aligned.csv"

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
SOURCE_GREENHOUSE = "alava"
DESTINATION_GREENHOUSE = "cartagena"
# JSON files are named after this script: <script name>_<condition>.json
SCRIPT_ID = os.path.splitext(os.path.basename(__file__))[0]

RESULTS_DIR = f"./results/finetuning/{SOURCE_GREENHOUSE}_to_{DESTINATION_GREENHOUSE}"
# Parquet predictions: PREDICTIONS_DIR/<zero_shot|finetuned|reference>/<target_dir>/
PREDICTIONS_DIR = f"./results/predictions/finetuning/{SOURCE_GREENHOUSE}_to_{DESTINATION_GREENHOUSE}"

TARGET_DIR_NAMES = {
    "DB_TEMP_INT": "temperature",
    "REL_HUM_INT": "rel_humidity",
}

for _target_dir in TARGET_DIR_NAMES.values():
    os.makedirs(os.path.join(RESULTS_DIR, _target_dir), exist_ok=True)


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
# PHASE 0: pre-training on Alava (from scratch)
# =============================================================================
def prepare_and_pretrain(df1, df2):
    # --- 70/30 split on each CSV ---
    split_70_csv1 = int(len(df1) * 0.7)
    df1_trainval = df1.iloc[:split_70_csv1].copy()

    split_70_csv2 = int(len(df2) * 0.7)
    df2_trainval = df2.iloc[:split_70_csv2].copy()
    df2_test = df2.iloc[split_70_csv2:].copy()

    # --- Alava train/val blocks (pre-training) ---
    source_labels, source_block_id = label_day_blocks(df1_trainval, train_block_days, val_block_days)
    source_blocks = split_into_blocks(df1_trainval, source_labels, source_block_id)
    source_train_blocks = [sub for label, sub in source_blocks if label == "train"]
    source_val_blocks = [sub for label, sub in source_blocks if label == "val"]

    print(f"[{csv1_name}] train blocks: {len(source_train_blocks)} | val: {len(source_val_blocks)}")

    # --- Cartagena train/val blocks (fine-tuning) ---
    destination_labels, destination_block_id = label_day_blocks(df2_trainval, train_block_days, val_block_days)
    destination_blocks = split_into_blocks(df2_trainval, destination_labels, destination_block_id)
    destination_train_blocks = [sub for label, sub in destination_blocks if label == "train"]
    destination_val_blocks = [sub for label, sub in destination_blocks if label == "val"]

    print(f"[{csv2_name}] train blocks (fine-tuning): {len(destination_train_blocks)} | "
          f"val (fixed): {len(destination_val_blocks)} | test (fixed, 30%): {len(df2_test)} rows")

    available_destination_days = sum(len(b) for b in destination_train_blocks) / samples_per_day
    print(f"[{csv2_name}] available fine-tuning train days: ~{available_destination_days:.1f}")

    # --- scaler fit ONLY on Alava's train ---
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

    print(f"Pre-training from scratch on {csv1_name}...")
    source_X_train, source_y_train = generate_sequences_by_blocks(
        scaled_source_train_blocks, variate_columns, window_size, horizon, stride=stride_train
    )
    source_X_val, source_y_val = generate_sequences_by_blocks(
        scaled_source_val_blocks, variate_columns, window_size, horizon, stride=stride_val
    )
    source_train_loader = DataLoader(ITransformerDataset(source_X_train, source_y_train), batch_size=batch_size, shuffle=True)
    source_val_loader = DataLoader(ITransformerDataset(source_X_val, source_y_val), batch_size=batch_size, shuffle=False)

    pretrain_model = create_model(n_vars, window_size, horizon, cfg_modelo, device)

    inicio = time.time()
    _, pretrained_state, best_epoch, _, _ = train(
        pretrain_model, source_train_loader, source_val_loader, device,
        lr=lr_pretrain, num_epochs=num_epochs_pretrain, patience=patience_pretrain,
        mostrar_barra=True, log_prefix=f"[Pretrain-{csv1_name}] ",
    )
    print(f"Pre-training finished in {time.time() - inicio:.1f}s (best epoch: {best_epoch})")

    context = {
        "n_vars": n_vars,
        "scaled_destination_train_blocks": scaled_destination_train_blocks,
        "scaled_destination_val_blocks": scaled_destination_val_blocks,
        "scaled_destination_test_df": scaled_destination_test_df,
        "pretrained_state": pretrained_state,
        "available_destination_days": available_destination_days,
        "scaler_X": scaler_X,
        # Unscaled destination data, for the same-greenhouse reference model
        "destination_train_blocks": destination_train_blocks,
        "destination_val_blocks": destination_val_blocks,
        "destination_test_df": df2_test,
    }
    return context


# =============================================================================
# EXPERIMENT C: direct comparison (zero-shot transfer vs fine-tuning)
# =============================================================================
def evaluate_full_dataset(model, df_test_esc, variate_columns, target_idx,
                          window_size, horizon, scaler_X, device, step=None):
    """
    Full evaluation, same scheme as 10_itransformer_alava.py
    (global + per_horizon + per_month + day/night), in actual units.

    Returns:
      metrics_by_target: {target: metrics dict}
      eval_dfs: {target: DataFrame indexed by date with columns
                 t0, actual, pred, horizon} (long format, same as the
                 parquet files saved by the base scripts)
    """
    model.eval()
    step = step or horizon

    rows_by_target = {name: [] for name in target_idx}

    with torch.no_grad():
        for i in range(0, len(df_test_esc) - window_size - horizon + 1, step):
            x_in = df_test_esc[variate_columns].iloc[i:i + window_size].values
            x_in_t = torch.tensor(x_in, dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(x_in_t).cpu().numpy()[0]  # (horizon, n_vars), scaled
            actual_block = df_test_esc[variate_columns].iloc[
                i + window_size: i + window_size + horizon
            ].values  # (horizon, n_vars), scaled

            pred_desc = scaler_X.inverse_transform(pred)
            actual_desc = scaler_X.inverse_transform(actual_block)

            base_time = df_test_esc.index[i + window_size]
            for h in range(horizon):
                date = base_time + pd.Timedelta(minutes=15 * (h + 1))
                for name, idx in target_idx.items():
                    rows_by_target[name].append({
                        "date": date,
                        "t0": base_time,
                        "actual": actual_desc[h, idx],
                        "pred": pred_desc[h, idx],
                        "horizon": h + 1,
                    })

    metrics_by_target = {}
    eval_dfs = {}
    for name, rows in rows_by_target.items():
        df_eval = pd.DataFrame(rows).set_index("date").sort_index()
        eval_dfs[name] = df_eval

        metrics = {
            "global": calculate_metrics(df_eval["actual"], df_eval["pred"]),
            "per_horizon": {},
            "per_month": {},
            "day": {},
            "night": {},
        }

        day_mask = np.array([
            is_it_day_from_date(date.to_pydatetime(), LAT, LON) for date in df_eval.index
        ])
        if day_mask.sum() >= 2:
            metrics["day"] = calculate_metrics(df_eval.loc[day_mask, "actual"], df_eval.loc[day_mask, "pred"])
        if (~day_mask).sum() >= 2:
            metrics["night"] = calculate_metrics(df_eval.loc[~day_mask, "actual"], df_eval.loc[~day_mask, "pred"])

        for h in range(1, horizon + 1):
            g_h = df_eval[df_eval["horizon"] == h]
            if len(g_h) < 2:
                continue
            metrics["per_horizon"][f"h_{h}"] = calculate_metrics(g_h["actual"], g_h["pred"])

        for month, group in df_eval.groupby(pd.Grouper(freq="ME")):
            if len(group) < 2:
                continue
            metrics["per_month"][month.strftime("%Y-%m")] = calculate_metrics(group["actual"], group["pred"])

        metrics_by_target[name] = metrics

    return metrics_by_target, eval_dfs


def save_predictions_parquet(eval_dfs, condition):
    """
    Saves one parquet per target under
    PREDICTIONS_DIR/<condition>/<target_dir>/ and returns
    {target: parquet path}.
    """
    prediction_paths = {}
    for name, df_eval in eval_dfs.items():
        predictions_dir = os.path.join(PREDICTIONS_DIR, condition, TARGET_DIR_NAMES[name])
        os.makedirs(predictions_dir, exist_ok=True)
        path = os.path.join(predictions_dir, f"exp_itransformer_{name}_{uuid.uuid4().hex}.parquet")
        df_eval.to_parquet(path)
        prediction_paths[name] = path
    return prediction_paths


def save_reference_style_json(filepath, target_variable, metrics, predictions=None, notes=""):
    """
    Same pattern as the base script's `save_experiment_json`: if the file
    already exists, APPENDS a new entry to the list (does not overwrite).
    """
    experiment = {
        "target_variable": target_variable,
        "metrics": metrics,
        "predictions": predictions,
        "notes": notes,
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
    else:
        data = []

    data.append(experiment)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4, default=str)


def train_same_greenhouse_model(context):
    """
    Trains a model from scratch only on the destination greenhouse
    (its interleaved train/val blocks, same input variables), with its own
    MinMaxScaler fit on the destination train blocks, and evaluates it on
    the destination's fixed test set.

    Returns:
      metrics_by_target, eval_dfs, best_epoch
    """
    scaler_same = MinMaxScaler()
    scaler_same.fit(pd.concat(context["destination_train_blocks"])[variate_columns])

    def scale(sub_df):
        sub_df = sub_df.copy()
        sub_df[variate_columns] = scaler_same.transform(sub_df[variate_columns])
        return sub_df

    train_blocks = [scale(b) for b in context["destination_train_blocks"]]
    val_blocks = [scale(b) for b in context["destination_val_blocks"]]
    test_df = scale(context["destination_test_df"])

    X_train, y_train = generate_sequences_by_blocks(
        train_blocks, variate_columns, window_size, horizon, stride=stride_train
    )
    X_val, y_val = generate_sequences_by_blocks(
        val_blocks, variate_columns, window_size, horizon, stride=stride_val
    )
    train_loader = DataLoader(ITransformerDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ITransformerDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    print(f"Training same-greenhouse model from scratch on {csv2_name} ({len(y_train)} sequences)...")
    model = create_model(context["n_vars"], window_size, horizon, cfg_modelo, device)

    inicio = time.time()
    _, best_state, best_epoch, _, _ = train(
        model, train_loader, val_loader, device,
        lr=lr_pretrain, num_epochs=num_epochs_pretrain, patience=patience_pretrain,
        mostrar_barra=True, log_prefix="[Comparison-same-greenhouse] ",
    )
    print(f"Same-greenhouse training finished in {time.time() - inicio:.1f}s (best epoch: {best_epoch})")
    model.load_state_dict(best_state)

    metrics, eval_dfs = evaluate_full_dataset(
        model, test_df, variate_columns, target_idx,
        window_size, horizon, scaler_same, device,
    )
    return metrics, eval_dfs, best_epoch


def direct_comparison_experiment(
    context,
    finetune_days=None,   # None = use 100% of the available fine-tuning days
    num_epochs=100,
    patience=10,
    zeroshot_suffix="zeroshot",
    finetuned_suffix="finetuned",
    same_greenhouse_suffix="reference",
):
    """
    Generates, for each target, three JSON files:

      RESULTS_DIR/<target_dir>/<SCRIPT_ID>_zeroshot.json
        -> pre-trained model WITHOUT fine-tuning, evaluated on csv_2's test.

      RESULTS_DIR/<target_dir>/<SCRIPT_ID>_finetuned.json
        -> the same model fine-tuned on `finetune_days` days of csv_2,
           evaluated on the same fixed test set.

      RESULTS_DIR/<target_dir>/<SCRIPT_ID>_reference.json
        -> reference model trained from scratch only on csv_2 (same input
           variables, own scaler), evaluated on the same fixed test set.

    The predictions of each condition are also saved as parquet (one per
    target) under PREDICTIONS_DIR/zero_shot/<target_dir>/,
    PREDICTIONS_DIR/finetuned/<target_dir>/ and
    PREDICTIONS_DIR/reference/<target_dir>/,
    and their path is stored in the "predictions" field of the JSON.
    """
    print("\n=== EXPERIMENT C: direct comparison (zero-shot vs fine-tuning vs same greenhouse) ===")

    destination_val_loader = DataLoader(
        ITransformerDataset(*generate_sequences_by_blocks(
            context["scaled_destination_val_blocks"], variate_columns, window_size, horizon, stride=stride_val
        )),
        batch_size=batch_size, shuffle=False,
    )

    # --- 1) Zero-shot ---
    print("Evaluating direct transfer (zero-shot, no fine-tuning)...")
    zeroshot_model = create_model(context["n_vars"], window_size, horizon, cfg_modelo, device)
    zeroshot_model.load_state_dict(context["pretrained_state"])
    zeroshot_metrics, zeroshot_eval_dfs = evaluate_full_dataset(
        zeroshot_model, context["scaled_destination_test_df"], variate_columns, target_idx,
        window_size, horizon, context["scaler_X"], device,
    )
    zeroshot_prediction_paths = save_predictions_parquet(zeroshot_eval_dfs, "zero_shot")

    # --- 2) Fine-tuning ---
    sub_blocks = select_first_days(context["scaled_destination_train_blocks"], finetune_days, samples_per_day)
    X_sub, y_sub = generate_sequences_by_blocks(sub_blocks, variate_columns, window_size, horizon, stride=stride_train)
    train_loader = DataLoader(ITransformerDataset(X_sub, y_sub), batch_size=batch_size, shuffle=True)

    days_label = finetune_days if finetune_days is not None else f"{context['available_destination_days']:.0f}(all)"
    print(f"Fine-tuning with {days_label} days ({len(y_sub)} sequences)...")

    finetuned_model = create_model(context["n_vars"], window_size, horizon, cfg_modelo, device)
    finetuned_model.load_state_dict(context["pretrained_state"])

    _, best_state, best_epoch, _, _ = train(
        finetuned_model, train_loader, destination_val_loader, device,
        lr=lr_finetune, num_epochs=num_epochs, patience=patience,
        log_prefix="[Comparison-finetuning] ",
    )
    finetuned_model.load_state_dict(best_state)

    finetuned_metrics, finetuned_eval_dfs = evaluate_full_dataset(
        finetuned_model, context["scaled_destination_test_df"], variate_columns, target_idx,
        window_size, horizon, context["scaler_X"], device,
    )
    finetuned_prediction_paths = save_predictions_parquet(finetuned_eval_dfs, "finetuned")

    # --- 3) Same greenhouse (reference) ---
    same_metrics, same_eval_dfs, same_best_epoch = train_same_greenhouse_model(context)
    same_prediction_paths = save_predictions_parquet(same_eval_dfs, "reference")

    # --- 4) Save JSON + console summary ---
    print(f"\n{'target':<14}{'condition':<17}{'r2':>8}{'rmse':>10}{'mae':>10}")
    for target_label in target_idx:
        target_results_dir = os.path.join(RESULTS_DIR, TARGET_DIR_NAMES[target_label])

        zeroshot_path = os.path.join(target_results_dir, f"{SCRIPT_ID}_{zeroshot_suffix}.json")
        save_reference_style_json(
            zeroshot_path, target_label, zeroshot_metrics[target_label],
            predictions=zeroshot_prediction_paths[target_label],
            notes=(f"Direct transfer (zero-shot): model trained only on "
                   f"'{csv1_name}', evaluated with no fine-tuning on the test set "
                   f"of '{csv2_name}'."),
        )

        finetuned_path = os.path.join(target_results_dir, f"{SCRIPT_ID}_{finetuned_suffix}.json")
        save_reference_style_json(
            finetuned_path, target_label, finetuned_metrics[target_label],
            predictions=finetuned_prediction_paths[target_label],
            notes=(f"Fine-tuning: model pre-trained on '{csv1_name}' and then "
                   f"fine-tuned with {days_label} days of '{csv2_name}' "
                   f"(best val epoch: {best_epoch}); evaluated on the same fixed test set."),
        )

        same_path = os.path.join(target_results_dir, f"{SCRIPT_ID}_{same_greenhouse_suffix}.json")
        save_reference_style_json(
            same_path, target_label, same_metrics[target_label],
            predictions=same_prediction_paths[target_label],
            notes=(f"Same greenhouse (reference): model trained from scratch only on "
                   f"'{csv2_name}' with the same input variables and its own scaler "
                   f"(best val epoch: {same_best_epoch}); evaluated on the same fixed test set."),
        )

        g_zs = zeroshot_metrics[target_label]["global"]
        g_ft = finetuned_metrics[target_label]["global"]
        g_sg = same_metrics[target_label]["global"]
        print(f"{target_label:<14}{'zero-shot':<17}{g_zs['r2']:>8.3f}{g_zs['rmse']:>10.3f}{g_zs['mae']:>10.3f}")
        print(f"{target_label:<14}{'finetuned':<17}{g_ft['r2']:>8.3f}{g_ft['rmse']:>10.3f}{g_ft['mae']:>10.3f}")
        print(f"{target_label:<14}{'same greenhouse':<17}{g_sg['r2']:>8.3f}{g_sg['rmse']:>10.3f}{g_sg['mae']:>10.3f}")

    print(f"\nJSON saved under {RESULTS_DIR}/<target>/ as {SCRIPT_ID}_<condition>.json "
          f"(one per target x condition).")
    print(f"Parquet predictions saved under {PREDICTIONS_DIR}/<zero_shot|finetuned|reference>/<target>/.")

    return zeroshot_metrics, finetuned_metrics, same_metrics


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    df1, df2 = load_datasets()
    context = prepare_and_pretrain(df1, df2)

    direct_comparison_experiment(context, finetune_days=None, num_epochs=100, patience=10)

    print(f"\nDone. Experiment C ('{RESULTS_DIR}/temperature/', "
          f"'{RESULTS_DIR}/rel_humidity/') is ready for the comparison notebook.")