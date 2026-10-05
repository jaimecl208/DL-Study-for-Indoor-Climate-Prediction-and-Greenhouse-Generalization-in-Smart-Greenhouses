"""
iTransformer (Liu et al., 2024, ICLR) adapted to predicting indoor
greenhouse temperature / relative humidity.

Faithfulness to the paper:
  - No decoder, no mechanism for "known-in-the-future" covariates: the
    original paper does NOT distinguish between observed and known
    variables. All input variables (including the calendar features
    month_sin, month_cos, is_day) are treated as just more channels,
    exactly as in the previous PatchTST usage.
  - INVERTED embedding: each variable, with its whole temporal window
    (window_size steps), is projected to a single token of dimension
    d_model (nn.Linear(window_size, d_model)). That is, the tokens are
    variables, not time steps (hence the name "inverted").
  - Standard Transformer encoder (multi-head self-attention + position-
    wise feed-forward + Add&Norm), applied over the variable dimension:
    attention learns relationships BETWEEN variables, not between time
    steps. No causal mask (it wouldn't make sense: attention is not over
    time).
  - Instance normalization (subtracting the mean / dividing by the std of
    each window, not learnable) before the embedding, and de-
    normalization at the output. This is the Non-stationary-Transformer-
    style normalization that the official iTransformer repo uses by
    default (the `use_norm` flag).
  - Final linear projection (d_model -> horizon) per variable: predicts
    the whole horizon for all variables in a single pass (standard
    multivariate forecasting, no autoregression).
  - Point-wise output (not quantile-based) and MSE as the loss, exactly
    as in the paper.

Reuses the same data pipeline (interleaved train/val block split,
MinMaxScaler scaling, evaluation and JSON experiment saving) as the
previous models, so comparisons are fair.

Since the model predicts ALL input variates in a single pass, temperature
and indoor relative humidity are obtained from the SAME model trained
only once: there is no need to train two separate models. Two experiment
JSONs are saved (one per target), reusing the same `model`, just as in the
PatchTST script.
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
INPUT_CSV = "./data/processed/Alava_processed.csv"
INVERNADERO = "alava"
MODEL_NAME = "itransformer"
SCRIPT_ID = "10_itransformer_alava"

# Folder name used per target under results/<predictions|experiments>/<greenhouse>/<model>/<target_dir>/
TARGET_DIR_NAMES = {
    "DB_TEMP_INT": "temperature",
    "REL_HUM_INT": "rel_humidity",
}

LAT = 42.853615
LON = -2.674332


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
class ITransformerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------
# Multivariate sequences (past window -> future window, ALL variables,
# without separating obs/known)
# ---------------------------
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


# ---------------------------
# iTransformer building blocks
# ---------------------------
class InvertedEmbedding(nn.Module):
    """
    Inverted embedding: each variable (channel), with its full window_size
    temporal window, is projected to a token of dimension d_model. The
    result is N tokens (one per variable), not L tokens (one per time
    step), which is the inversion relative to a Transformer applied to
    time series in the standard way.
    """
    def __init__(self, window_size, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(window_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, window_size, n_vars) -> (batch, n_vars, window_size)
        x = x.permute(0, 2, 1)
        return self.dropout(self.value_embedding(x))  # (batch, n_vars, d_model)


class EncoderLayer(nn.Module):
    """Standard Transformer encoder layer: self-attention + position-wise FFN + Add&Norm."""
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
        # Self-attention BETWEEN variables (tokens = variables), no causal mask
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


class Modelo(nn.Module):
    """
    Full iTransformer: instance normalization -> inverted embedding ->
    encoder (attention between variables) -> linear projection to
    `horizon` steps per variable -> de-normalization.
    Predicts ALL input variables for the whole horizon in a single pass
    (the paper's standard multivariate forecasting).
    """
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
        # x: (batch, window_size, n_vars)
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x = x - means
            stds = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / stds

        tokens = self.embedding(x)          # (batch, n_vars, d_model)
        enc_out, _ = self.encoder(tokens)    # (batch, n_vars, d_model)
        dec_out = self.projector(enc_out)    # (batch, n_vars, horizon)
        dec_out = dec_out.permute(0, 2, 1)   # (batch, horizon, n_vars)

        if self.use_norm:
            dec_out = dec_out * stds.repeat(1, dec_out.size(1), 1) + means.repeat(1, dec_out.size(1), 1)

        return dec_out


# ---------------------------
# Config
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128
lr = 0.0001
num_epochs = 100
patience = 10
window_size = 96
horizon = 24

# iTransformer-specific hyperparameters
# (d_model/d_ff smaller than the typical 512 in the paper because the
# original benchmarks use much larger datasets; adjust if needed)
d_model = 128
n_heads = 8
e_layers = 3
d_ff = 128
dropout = 0.1
use_norm = False  # instance normalization, as in the official repo

# Interleaved train/val blocks (identical to the previous models)
train_block_days = 14
val_block_days = 7
stride_train = 1
stride_val = 1

# All variables enter as channels of the multivariate forecasting, with no
# observed/known distinction (iTransformer does not model that
# distinction). The model is trained and predicts on ALL variates at
# once; from that multivariate output we keep two columns to evaluate
# (indoor temperature and humidity), just as in PatchTST.
variate_columns =  [
    'DB_TEMP_INT_INPUT_Alava',
    "DB_TEMP_EXT_Alava",
    "REL_HUM_EXT_Alava",
    "RADIATION_MEAN_Alava",
    "REL_HUM_INT_INPUT_Alava",
    'PIPE_TEMP_Alava',
    'month_sin',
    'month_cos',
    'is_day',

]

# The thesis's two targets. They must be among the input variates
# (autoregression on their own past); if they aren't already (e.g.
# because only the "INPUT" or "lag" variant is present as a feature),
# they get added.
target_columns = {
    "DB_TEMP_INT": "DB_TEMP_INT_INPUT_Alava",
    "REL_HUM_INT": "REL_HUM_INT_INPUT_Alava",
}

for col in target_columns.values():
    if col not in variate_columns:
        variate_columns.append(col)

target_idx = {name: variate_columns.index(col) for name, col in target_columns.items()}


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
# A single scaler for ALL variates (X and y live in the same variable
# space, as in PatchTST): each channel predicts/inverse-transforms its
# own future with the same scaler.
# ---------------------------
scaler_X = MinMaxScaler()

df_train_concat = pd.concat(train_blocks)
scaler_X.fit(df_train_concat[variate_columns])

def scale_block(sub_df):
    sub_df = sub_df.copy()
    sub_df[variate_columns] = scaler_X.transform(sub_df[variate_columns])
    return sub_df

scaled_train_blocks = [scale_block(b) for b in train_blocks]
scaled_val_blocks = [scale_block(b) for b in val_blocks]

df_test[variate_columns] = scaler_X.transform(df_test[variate_columns])


# ---------------------------
# Sequences
# ---------------------------
X_train, y_train = generate_sequences_by_blocks(
    scaled_train_blocks, variate_columns, window_size, horizon, stride=stride_train
)
X_val, y_val = generate_sequences_by_blocks(
    scaled_val_blocks, variate_columns, window_size, horizon, stride=stride_val
)
X_test, y_test = prepare_multivariate_sequences(
    df_test, variate_columns, window_size, horizon
)

print(f"Sequences -> train: {len(y_train)} | val: {len(y_val)} | test: {len(y_test)}")


# ---------------------------
# DataLoaders
# ---------------------------
train_loader = DataLoader(
    ITransformerDataset(X_train, y_train),
    batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    ITransformerDataset(X_val, y_val),
    batch_size=batch_size, shuffle=False
)


# ---------------------------
# Model
# ---------------------------
n_vars = len(variate_columns)

model = Modelo(
    n_vars=n_vars,
    window_size=window_size,
    horizon=horizon,
    d_model=d_model,
    n_heads=n_heads,
    e_layers=e_layers,
    d_ff=d_ff,
    dropout=dropout,
    use_norm=use_norm,
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

    for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]", leave=False):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            outputs = model(x_batch)
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
# The model predicts ALL variates in a single pass; from that multivariate
# output we extract the two target columns (indoor temperature and
# humidity), just as in the PatchTST script. Rows in the same long format
# (date, t0, actual, pred, horizon) as the rest of the scripts, one
# DataFrame per target.
# ---------------------------
model.eval()

rows_by_target = {name: [] for name in target_columns}
step = horizon

with torch.no_grad():

    for i in range(0, len(df_test) - window_size - horizon + 1, step):

        x_in = df_test[variate_columns].iloc[i:i + window_size].values
        x_in_t = torch.tensor(x_in, dtype=torch.float32).unsqueeze(0).to(device)

        pred = model(x_in_t).cpu().numpy()[0]  # (horizon, n_vars), scaled

        actual_block = df_test[variate_columns].iloc[
            i + window_size: i + window_size + horizon
        ].values  # (horizon, n_vars), scaled

        # Inverse-transform with the single scaler, over the full row (horizon, n_vars)
        pred_desc = scaler_X.inverse_transform(pred)        # (horizon, n_vars)
        actual_desc = scaler_X.inverse_transform(actual_block)  # (horizon, n_vars)

        base_time = df_test.index[i + window_size]

        for h in range(horizon):
            date = base_time + pd.Timedelta(minutes=15 * (h + 1))
            for name, idx in target_idx.items():
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
    path = os.path.join(predictions_dir, f"exp_itransformer_{name}_{uuid.uuid4().hex}.parquet")
    df_eval_t.to_parquet(path)
    prediction_paths[name] = path


# ---------------------------
# Metrics helper function (same schema as PatchTST)
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
# Full metrics for each target (global / day / night / per_horizon / per_month)
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
# Save experiment
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
    "architecture": "iTransformer (Liu et al., 2024)",
    "batch_size": batch_size,
    "lr": lr,
    "num_epochs": num_epochs,
    "patience": patience,
    "window_size": window_size,
    "horizon": horizon,
    "d_model": d_model,
    "n_heads": n_heads,
    "e_layers": e_layers,
    "d_ff": d_ff,
    "dropout": dropout,
    "use_norm": use_norm,
    "train_block_days": train_block_days,
    "val_block_days": val_block_days,
    "stride_train": stride_train,
    "stride_val": stride_val,
}

input_variables = {
    "variates": variate_columns,
}
notes = (
    "iTransformer faithful to the original paper: inverted embedding "
    "(one token per variable), encoder with self-attention between "
    "variables, instance normalization, linear projection to horizon, "
    "point-wise output with MSE. No obs/known distinction: "
    "month_sin/month_cos/is_day enter as just more variates, as in the "
    "previous PatchTST usage. Trained and evaluated on all variates; an "
    "independent JSON is saved per target (DB_TEMP_INT and REL_HUM_INT), "
    "reusing the same model."
)

# One JSON per target, with exactly the same schema for both
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
