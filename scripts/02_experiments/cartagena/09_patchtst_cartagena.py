"""
PatchTST faithful to the original paper ("A Time Series is Worth 64 Words")
applied to predicting indoor greenhouse variables.

Design (same as the paper):
- Each channel (variable) is split into patches and passed through the
  SAME Transformer encoder (weights shared across channels).
- Each channel predicts ITS OWN future with a linear head (flatten +
  linear) also with weights shared across channels. There is no point in
  the model where information is mixed between channels: "channel
  independence" is maintained end to end.
- The problem is therefore multivariate -> multivariate: every input
  variable is predicted `horizon` steps ahead, each one using only its
  own past.

Of all the predicted channels, only two are of interest for this thesis:
DB_TEMP_INT and REL_HUM_INT. Metrics are computed only on those two
channels (selected by their index in `input_columns`), even though the
model is trained and predicts on every input channel.

Reuses the same data pipeline (interleaved train/val block split,
MinMaxScaler scaling, evaluation and JSON experiment saving) as the base
LSTM model, so comparisons are fair.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import numpy as np
import pandas as pd
import math

from tqdm import tqdm
import time
import os
import uuid
import json
import inspect

import ephem

# ---------------------------
# Config: paths
# ---------------------------
INPUT_CSV = "./data/processed/Cartagena_comp0_processed.csv"
INVERNADERO = "cartagena"
MODEL_NAME = "patchtst"
SCRIPT_ID = "09_patchtst_cartagena"

# Folder name used per target under results/<predictions|experiments>/<greenhouse>/<model>/<target_dir>/
TARGET_DIR_NAMES = {
    "DB_TEMP_INT": "temperature",
    "REL_HUM_INT": "rel_humidity",
}

LAT = 42.8526
LON = -2.62629


def is_it_day_from_date(date, lat, lon):
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.date = date

    sun = ephem.Sun(observer)

    return observer.previous_rising(sun) > observer.previous_setting(sun)


# ---------------------------
# Dataset
# ---------------------------
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------
# Sequences
# ---------------------------
def prepare_sequences(df, columns, window_size, horizon, stride=1):
    """
    Generates multivariate -> multivariate sequences.

    X: (n_samples, window_size, n_vars)   -- past window, all channels
    y: (n_samples, n_vars, horizon)       -- each channel's future,
                                              already shaped (n_vars,
                                              horizon) as produced by the
                                              model, to avoid permuting
                                              later.
    """
    X, y = [], []

    data = df[columns].values

    limite = len(df) - window_size - horizon + 1
    for i in range(0, max(limite, 0), stride):
        X.append(data[i:i + window_size])
        # (horizon, n_vars) -> (n_vars, horizon)
        y.append(data[i + window_size:i + window_size + horizon].T)

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


def generate_sequences_by_blocks(blocks, input_columns, window_size, horizon, stride):
    Xs, ys = [], []
    for b in blocks:
        if len(b) < window_size + horizon:
            continue
        Xb, yb = prepare_sequences(b, input_columns, window_size, horizon, stride=stride)
        if len(Xb) > 0:
            Xs.append(Xb)
            ys.append(yb)

    if not Xs:
        raise ValueError("Could not generate any sequence from these blocks.")

    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


# ---------------------------
# PatchTST model
# ---------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, d_model, window_size):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (window_size - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x):
        # x: (batch, n_vars, window_size)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: (batch, n_vars, num_patches, patch_len)
        x = self.proj(x)
        # x: (batch, n_vars, num_patches, d_model)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, num_patches, d_model)
        return x + self.pe[:x.size(1)]


class Modelo(nn.Module):
    """
    Channel-independent PatchTST, faithful to the original paper:
    - patch embedding and encoder with weights shared across channels.
    - linear head (flatten + linear) ALSO with weights shared across
      channels, applied BEFORE undoing the batch/channel merge. Each
      channel predicts its own future without seeing information from
      the others.
    """
    def __init__(self, n_vars, window_size, horizon, patch_len=4, stride=2,
                 d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()

        self.n_vars = n_vars
        self.horizon = horizon
        self.patch_embedding = PatchEmbedding(patch_len, stride, d_model, window_size)
        self.num_patches = self.patch_embedding.num_patches
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_patches + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.dropout = nn.Dropout(dropout)
        # Head shared across channels: receives (num_patches * d_model)
        # and predicts `horizon` steps for ONE channel.
        self.head = nn.Linear(self.num_patches * d_model, horizon)

    def forward(self, x):
        # x: (batch, window_size, n_vars) -> (batch, n_vars, window_size)
        x = x.permute(0, 2, 1)
        batch = x.size(0)

        x = self.patch_embedding(x)  # (batch, n_vars, num_patches, d_model)

        # channel independence: merge batch and n_vars to pass through the
        # same encoder (weights shared across channels)
        x = x.reshape(batch * self.n_vars, self.num_patches, -1)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = self.dropout(x)

        # shared head applied channel by channel, WITHOUT merging channels
        # back together (channel independence end to end)
        x = x.reshape(batch * self.n_vars, -1)          # (batch*n_vars, num_patches*d_model)
        out = self.head(x)                               # (batch*n_vars, horizon)
        out = out.reshape(batch, self.n_vars, self.horizon)  # (batch, n_vars, horizon)
        return out


# ---------------------------
# Metrics
# ---------------------------
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


# ---------------------------
# Config
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128
lr = 0.0001
num_epochs = 100
patience = 10
window_size = 96*3
horizon = 24

# PatchTST-specific hyperparameters
patch_len = 24
patch_stride = 12
d_model = 64
nhead = 4
num_encoder_layers = 3
dim_feedforward = 128
dropout = 0.1

# Interleaved train/val blocks (identical to the base LSTM model)
train_block_days = 14
val_block_days = 7
stride_train = 1
stride_val = 1

input_columns = [
    'DB_TEMP_INT_INPUT_Cartagena_comp0',
    "DB_TEMP_EXT_Cartagena_meteo",
    "REL_HUM_EXT_Cartagena_meteo",
    "RADIATION_MEAN_Cartagena_meteo",
    "REL_HUM_INT_Cartagena_comp0",
    'month_sin',
    'month_cos',
    'is_day',
    'WB_TEMP_Cartagena_comp0',
    'VPD_Cartagena_comp0',
    'WATER_DEF_Cartagena_comp0',
]

# Target channels for this thesis (subset of input_columns, since the
# model predicts the future of ALL channels, but only these two are
# evaluated). 'DB_TEMP_INT_INPUT_Cartagena_comp0' holds the same values as
# 'DB_TEMP_INT_Cartagena_comp0' (duplicated only so as not to break the
# scaling shared with the LSTM pipeline).
target_columns = {
    "DB_TEMP_INT": input_columns.index("DB_TEMP_INT_INPUT_Cartagena_comp0"),
    "REL_HUM_INT": input_columns.index("REL_HUM_INT_Cartagena_comp0"),
}


# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(
    INPUT_CSV,
    index_col=0,
    parse_dates=True
)

# ---------------------------
# Trainval / test split
# ---------------------------
split_70 = int(len(df) * 0.7)
df_trainval = df.iloc[:split_70].copy()
df_test = df.iloc[split_70:].copy()

# ---------------------------
# Train / val split by interleaved day blocks
# ---------------------------
labels, block_id = label_day_blocks(df_trainval, train_block_days, val_block_days)
blocks = split_into_blocks(df_trainval, labels, block_id)

train_blocks = [sub_df for label, sub_df in blocks if label == "train"]
val_blocks = [sub_df for label, sub_df in blocks if label == "val"]

print(f"Train blocks: {len(train_blocks)} | Val blocks: {len(val_blocks)}")
print(f"Train rows: {sum(len(b) for b in train_blocks)} | Val rows: {sum(len(b) for b in val_blocks)}")

# ---------------------------
# Scaling (fit only on train)
# ---------------------------
# A single scaler: X and y live in the same variable space
# (multivariate -> multivariate, each channel predicts its own future).
scaler = MinMaxScaler()

df_train_concat = pd.concat(train_blocks)
scaler.fit(df_train_concat[input_columns])

def scale_block(sub_df):
    sub_df = sub_df.copy()
    sub_df[input_columns] = scaler.transform(sub_df[input_columns])
    return sub_df

scaled_train_blocks = [scale_block(b) for b in train_blocks]
scaled_val_blocks = [scale_block(b) for b in val_blocks]

df_test[input_columns] = scaler.transform(df_test[input_columns])


# ---------------------------
# Sequences
# ---------------------------
X_train, y_train = generate_sequences_by_blocks(
    scaled_train_blocks, input_columns, window_size, horizon, stride=stride_train
)

X_val, y_val = generate_sequences_by_blocks(
    scaled_val_blocks, input_columns, window_size, horizon, stride=stride_val
)

X_test, y_test = prepare_sequences(
    df_test, input_columns, window_size, horizon
)

print(f"Sequences -> train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")


# ---------------------------
# DataLoaders
# ---------------------------
train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=batch_size, shuffle=False)


# ---------------------------
# Model
# ---------------------------
n_vars = len(input_columns)

model = Modelo(
    n_vars=n_vars,
    window_size=window_size,
    horizon=horizon,
    patch_len=patch_len,
    stride=patch_stride,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=4,
    min_lr=1e-6
)

# ---------------------------
# Early stopping
# ---------------------------
best_val_loss = float("inf")
counter = 0


# ---------------------------
# Training
# ---------------------------
inicio = time.time()

for epoch in range(num_epochs):

    model.train()
    train_loss = 0

    for X, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]", leave=False):
        X, y_batch = X.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X)                 # (batch, n_vars, horizon)
        loss = criterion(outputs, y_batch)  # loss over ALL channels, as in the paper
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X, y_batch in val_loader:
            X, y_batch = X.to(device), y_batch.to(device)

            outputs = model(X)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Epoch {epoch+1}: Train={train_loss:.4f} | Val={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

fin = time.time()
training_time = fin - inicio

# ---------------------------
# Evaluation
# ---------------------------
model.eval()

# Rows per target, in the same long format (date, t0, actual, pred,
# horizon) used by the base GRU/LSTM script, so the resulting JSON and
# parquet are 100% compatible with the existing analysis pipeline.
rows_by_target = {name: [] for name in target_columns}
step = horizon

with torch.no_grad():

    for i in range(0, len(df_test) - window_size - horizon + 1, step):

        xs = df_test[input_columns].iloc[i:i + window_size].values
        xs_t = torch.tensor(xs, dtype=torch.float32).unsqueeze(0).to(device)

        pred = model(xs_t).cpu().numpy()[0]  # (n_vars, horizon), scaled

        actual_block = df_test[input_columns].iloc[
            i + window_size: i + window_size + horizon
        ].values  # (horizon, n_vars), scaled

        # Inverse-transform with the single scaler, over (horizon, n_vars)
        pred_desc = scaler.inverse_transform(pred.T)       # (horizon, n_vars)
        actual_desc = scaler.inverse_transform(actual_block)   # (horizon, n_vars)

        base_time = df_test.index[i + window_size]

        for h in range(horizon):
            date = base_time + pd.Timedelta(minutes=15 * (h + 1))
            for name, idx in target_columns.items():
                rows_by_target[name].append({
                    "date": date,
                    "t0": base_time,
                    "actual": actual_desc[h, idx],
                    "pred": pred_desc[h, idx],
                    "horizon": h + 1
                })

eval_dfs = {}
prediction_paths = {}

for name, rows in rows_by_target.items():
    df_eval_t = pd.DataFrame(rows).set_index("date").sort_index()
    eval_dfs[name] = df_eval_t

    predictions_dir = f"./results/predictions/{INVERNADERO}/{MODEL_NAME}/{TARGET_DIR_NAMES[name]}"
    os.makedirs(predictions_dir, exist_ok=True)
    path = os.path.join(predictions_dir, f"exp_patchtst_{name}_{uuid.uuid4().hex}.parquet")
    df_eval_t.to_parquet(path)
    prediction_paths[name] = path


# ---------------------------
# Full metrics for each target channel (same schema as the base GRU/LSTM
# script: global / day / night / per_horizon / per_month)
# ---------------------------
metrics_by_target = {}

for name, df_eval in eval_dfs.items():

    y_true = df_eval["actual"].values
    y_pred = df_eval["pred"].values

    metrics = {
        "global": calculate_metrics(y_true, y_pred),
        "per_horizon": {},
        "per_month": {}
    }

    # Day / night
    metrics["day"] = {}
    metrics["night"] = {}

    mask_dia = np.array([
        is_it_day_from_date(date.to_pydatetime(), LAT, LON)
        for date in df_eval.index
    ])

    group = df_eval[mask_dia]
    if len(group) >= 2:
        metrics["day"] = calculate_metrics(group["actual"], group["pred"])

    group = df_eval[~mask_dia]
    if len(group) >= 2:
        metrics["night"] = calculate_metrics(group["actual"], group["pred"])

    # Per horizon
    for h in range(1, horizon + 1):
        g_h = df_eval[df_eval["horizon"] == h]
        if len(g_h) < 2:
            continue
        metrics["per_horizon"][f"h_{h}"] = calculate_metrics(g_h["actual"], g_h["pred"])

    # Per month
    for month, group in df_eval.groupby(pd.Grouper(freq="ME")):
        if len(group) < 2:
            continue
        metrics["per_month"][month.strftime("%Y-%m")] = calculate_metrics(group["actual"], group["pred"])

    metrics_by_target[name] = metrics

#----------------------
# Save experiment (same function and schema as the base GRU/LSTM script)
#----------------------

def save_experiment_json(
    filepath,
    input_variables,
    target_variable,
    model,
    optimizer,
    network_params,
    training_time,
    metrics,
    predictions,
    notes=None
):

    experiment = {
        "input_variables": input_variables,
        "target_variable": target_variable,
        "model_architecture": str(model),
        "forward_code": str(inspect.getsource(model.forward)),
        "optimizer": str(optimizer),
        "network_params": network_params,
        "training_time": training_time,
        "metrics": metrics,
        "predictions": predictions,
        "notes": notes if notes is not None else ""
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(experiment)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4, default=str)

network_params = {
    "architecture": "PatchTST",
    "batch_size": batch_size,
    "lr": lr,
    "num_epochs": num_epochs,
    "patience": patience,
    "window_size": window_size,
    "horizon": horizon,
    "patch_len": patch_len,
    "patch_stride": patch_stride,
    "d_model": d_model,
    "nhead": nhead,
    "num_encoder_layers": num_encoder_layers,
    "dim_feedforward": dim_feedforward,
    "dropout": dropout,
    "train_block_days": train_block_days,
    "val_block_days": val_block_days,
    "stride_train": stride_train,
    "stride_val": stride_val,
}

input_variables = input_columns
notes = (
    "Channel-independent PatchTST faithful to the original paper "
    "(multivariate -> multivariate, encoder and head shared across "
    "channels, no mixing of information between channels). Trained and "
    "evaluated on all input variables; an independent JSON is saved "
    "per target (DB_TEMP_INT and REL_HUM_INT), with the same schema as "
    "the base GRU/LSTM model, to keep the analysis pipeline compatible."
)

# One JSON per target, with exactly the same schema as gru_base.json
save_experiment_json(
    filepath=os.path.join(f"./results/experiments/{INVERNADERO}/{MODEL_NAME}/temperature", f"{SCRIPT_ID}.json"),
    input_variables=input_variables,
    target_variable="DB_TEMP_INT",
    model=model,
    optimizer=optimizer,
    network_params=network_params,
    training_time=training_time,
    metrics=metrics_by_target["DB_TEMP_INT"],
    predictions=prediction_paths["DB_TEMP_INT"],
    notes=notes
)

save_experiment_json(
    filepath=os.path.join(f"./results/experiments/{INVERNADERO}/{MODEL_NAME}/rel_humidity", f"{SCRIPT_ID}.json"),
    input_variables=input_variables,
    target_variable="REL_HUM_INT",
    model=model,
    optimizer=optimizer,
    network_params=network_params,
    training_time=training_time,
    metrics=metrics_by_target["REL_HUM_INT"],
    predictions=prediction_paths["REL_HUM_INT"],
    notes=notes
)
