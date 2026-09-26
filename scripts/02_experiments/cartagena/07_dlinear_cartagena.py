"""
Training and evaluation of DLinear as a univariate baseline.

Follows the same pipeline (day-block splits, sequence generation, metrics
and JSON saving format) as the GRU-based reference script for Cartagena,
so results are directly comparable across models.

Model design: each target variable is predicted EXCLUSIVELY from its own
history (univariate baseline, faithful to the original NLinear/DLinear
paper: "Are Transformers Effective for Time Series Forecasting?", Zeng et
al., 2022). DLinear decomposes the input window into trend + seasonality
(residual) via a moving average, and applies an independent linear layer
to each component.

The script runs once for each target variable defined in TARGETS, and
saves the results of each one to an independent JSON file.
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
INPUT_CSV = "./data/processed/Cartagena_comp0_processed.csv"
INVERNADERO = "cartagena"
MODEL_NAME = "dlinear"
SCRIPT_ID = "07_dlinear_cartagena"

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
class SecuenciaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------
# Model: DLinear
# ---------------------------
class MovingAvg(nn.Module):
    """Moving average used to extract the series' trend component."""

    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [batch, seq_len]
        # Pad by replicating the edges to preserve length after pooling
        front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
        end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=1)   # [batch, seq_len + kernel_size - 1]
        x = x.unsqueeze(1)                      # [batch, 1, seq_len + kernel_size - 1]
        x = self.avg(x)                         # [batch, 1, seq_len]
        x = x.squeeze(1)                        # [batch, seq_len]
        return x


class Modelo(nn.Module):
    """
    DLinear (Zeng et al., 2022), univariate baseline: predicts the target
    variable using only its own history.

    1) Decomposes the input window into trend (moving average) and
       seasonality/residual (what remains after subtracting the trend).
    2) Applies an independent linear layer to each component (seq_len -> horizon).
    3) Sums both predictions.
    """
    def __init__(self, seq_len, horizon, kernel_size=5):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.kernel_size = kernel_size

        self.moving_avg = MovingAvg(kernel_size, stride=1)
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        self.linear_trend = nn.Linear(seq_len, horizon)

    def forward(self, x):
        # x: [batch, seq_len, 1]
        x = x[:, :, 0]                     # [batch, seq_len]

        trend = self.moving_avg(x)         # [batch, seq_len]
        seasonal = x - trend               # [batch, seq_len]

        seasonal_out = self.linear_seasonal(seasonal)   # [batch, horizon]
        trend_out = self.linear_trend(trend)             # [batch, horizon]

        return seasonal_out + trend_out


# ---------------------------
# Sequences
# ---------------------------
def prepare_sequences(df, columns, output_column, window_size, horizon, stride=1):
    X, y = [], []

    data = df[columns].values
    y_data = df[output_column].values

    limite = len(df) - window_size - horizon + 1
    for i in range(0, max(limite, 0), stride):
        X.append(data[i:i + window_size])
        y.append(y_data[i + window_size:i + window_size + horizon])

    return np.array(X), np.array(y)


# ---------------------------
# Interleaved day blocks (train/val)
# ---------------------------
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


def generate_sequences_by_blocks(blocks, input_columns, output_column,
                                    window_size, horizon, stride):
    Xs, ys = [], []
    for b in blocks:
        if len(b) < window_size + horizon:
            continue
        Xb, yb = prepare_sequences(b, input_columns, output_column,
                                      window_size, horizon, stride=stride)
        if len(Xb) > 0:
            Xs.append(Xb)
            ys.append(yb)

    if not Xs:
        raise ValueError("Could not generate any sequence from these blocks.")

    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


# ---------------------------
# Experiment saving (same format as the reference script)
# ---------------------------
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


# ---------------------------
# General config (shared by both target variables)
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128
lr = 0.001
num_epochs = 100
patience = 10
window_size = 96
horizon = 24

# Moving-average window size used to separate trend/seasonality.
# Must be odd and coherent with window_size (here window_size=12 -> 3h
# windows at 15 min/step; adjust if you change the temporal resolution or
# the lookback).
kernel_size = 49

train_block_days = 3
val_block_days = 2
stride_train = 1
stride_val = 1

# Target variables to predict. Each one runs as an independent experiment
# (its own model, its own scaling, separate metrics and results JSON).
# Adjust the column names to match your CSV.
TARGETS = [
    {
        "output_column": "DB_TEMP_INT_Cartagena_comp0",
        "input_column": "DB_TEMP_INT_INPUT_Cartagena_comp0",
        "name": "temperatura",
        "target_dir": "temperature",
    },
    {
        "output_column": "REL_HUM_INT_Cartagena_comp0",
        "input_column": "REL_HUM_INT_INPUT_Cartagena_comp0",
        "name": "humedad_relativa",
        "target_dir": "rel_humidity",
    },
]


# ---------------------------
# Load data (once, reused for both target variables)
# ---------------------------
df = pd.read_csv(
    INPUT_CSV,
    index_col=0,
    parse_dates=True
)

# ---------------------------
# Trainval / test split (chronological: test = last segment, untouched)
# ---------------------------
split_70 = int(len(df) * 0.7)
df_trainval = df.iloc[:split_70].copy()
df_test_original = df.iloc[split_70:].copy()

# ---------------------------
# Train / val split by interleaved day blocks (independent of target)
# ---------------------------
labels, block_id = label_day_blocks(df_trainval, train_block_days, val_block_days)
blocks = split_into_blocks(df_trainval, labels, block_id)

raw_train_blocks = [sub_df for label, sub_df in blocks if label == "train"]
raw_val_blocks = [sub_df for label, sub_df in blocks if label == "val"]

print(f"Train blocks: {len(raw_train_blocks)} | Val blocks: {len(raw_val_blocks)}")
print(f"Train rows: {sum(len(b) for b in raw_train_blocks)} | Val rows: {sum(len(b) for b in raw_val_blocks)}")


# ---------------------------
# Runs the full pipeline (scaling, sequences, training, evaluation and
# saving) for ONE target variable
# ---------------------------
def ejecutar_experimento(output_column, input_column, target_dir, target_label):

    print("\n" + "=" * 60)
    print(f"Training DLinear for target variable: {output_column}")
    print("=" * 60)

    experiments_dir = f"./results/experiments/{INVERNADERO}/{MODEL_NAME}/{target_dir}"
    predictions_dir = f"./results/predictions/{INVERNADERO}/{MODEL_NAME}/{target_dir}"
    archivo_resultado = os.path.join(experiments_dir, f"{SCRIPT_ID}.json")

    # Univariate baseline: the only input is the target variable itself
    input_columns = [input_column]

    # ---------------------------
    # Scaling (fit only on this variable's train blocks)
    # ---------------------------
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    df_train_concat = pd.concat(raw_train_blocks)
    scaler_X.fit(df_train_concat[input_columns])
    scaler_y.fit(df_train_concat[[output_column]])

    def scale_block(sub_df):
        sub_df = sub_df.copy()
        sub_df[input_columns] = scaler_X.transform(sub_df[input_columns])
        sub_df[[output_column]] = scaler_y.transform(sub_df[[output_column]])
        return sub_df

    scaled_train_blocks = [scale_block(b) for b in raw_train_blocks]
    scaled_val_blocks = [scale_block(b) for b in raw_val_blocks]

    df_test = df_test_original.copy()
    df_test[input_columns] = scaler_X.transform(df_test[input_columns])
    df_test[[output_column]] = scaler_y.transform(df_test[[output_column]])

    # ---------------------------
    # Sequences
    # ---------------------------
    X_train, y_train = generate_sequences_by_blocks(
        scaled_train_blocks, input_columns, output_column, window_size, horizon, stride=stride_train
    )
    X_val, y_val = generate_sequences_by_blocks(
        scaled_val_blocks, input_columns, output_column, window_size, horizon, stride=stride_val
    )
    X_test, y_test = prepare_sequences(
        df_test, input_columns, output_column, window_size, horizon
    )

    print(f"Sequences -> train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")

    # ---------------------------
    # DataLoaders
    # ---------------------------
    train_loader = DataLoader(SecuenciaDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SecuenciaDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # ---------------------------
    # Model
    # ---------------------------
    model = Modelo(seq_len=window_size, horizon=horizon, kernel_size=kernel_size).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-6
    )

    best_val_loss = float("inf")
    counter = 0

    # ---------------------------
    # Training
    # ---------------------------
    inicio = time.time()

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0

        for X, y_batch in tqdm(train_loader, desc=f"[{target_label}] Epoch {epoch+1} [TRAIN]", leave=False):
            X, y_batch = X.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y_batch)
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

        print(f"[{target_label}] Epoch {epoch+1}: Train={train_loss:.4f} | Val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"[{target_label}] Early stopping triggered")
                break

    fin = time.time()
    training_time = fin - inicio

    # ---------------------------
    # Evaluation
    # ---------------------------
    model.eval()

    rows = []
    step = horizon

    with torch.no_grad():
        for i in range(0, len(df_test) - window_size - horizon + 1, step):

            xs = df_test[input_columns].iloc[i:i + window_size].values
            xs = torch.tensor(xs, dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(xs).cpu().numpy()[0]

            actual = df_test[output_column].iloc[
                i + window_size: i + window_size + horizon
            ].values

            pred = scaler_y.inverse_transform(pred.reshape(1, -1))[0]
            actual = scaler_y.inverse_transform(actual.reshape(1, -1))[0]

            base_time = df_test.index[i + window_size]

            for h in range(horizon):
                rows.append({
                    "date": base_time + pd.Timedelta(minutes=15 * (h + 1)),
                    "t0": base_time,
                    "actual": actual[h],
                    "pred": pred[h],
                    "horizon": h + 1
                })

    df_eval = pd.DataFrame(rows).set_index("date").sort_index()

    os.makedirs(predictions_dir, exist_ok=True)
    prediction_path = os.path.join(predictions_dir, f"dlinear_{target_label}_{uuid.uuid4().hex}.parquet")
    df_eval.to_parquet(prediction_path)

    # ---------------------------
    # Full metrics
    # ---------------------------
    y_true = df_eval["actual"].values
    y_pred = df_eval["pred"].values

    metrics = {
        "global": {
            "r2": float(r2_score(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100),
            "relative_error": float(np.mean((y_pred - y_true) / (y_true + 1e-8))),
            "relative_error_abs": float(np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8))))
        },
        "per_horizon": {},
        "per_month": {}
    }

    metrics["day"] = {}
    metrics["night"] = {}

    mask_dia = np.array([
        is_it_day_from_date(date.to_pydatetime(), LAT, LON)
        for date in df_eval.index
    ])

    group = df_eval[mask_dia]
    if len(group) >= 2:
        metrics["day"] = {
            "r2": float(r2_score(group["actual"], group["pred"])),
            "mse": float(mean_squared_error(group["actual"], group["pred"])),
            "rmse": float(np.sqrt(mean_squared_error(group["actual"], group["pred"]))),
            "mae": float(mean_absolute_error(group["actual"], group["pred"])),
            "mape": float(np.mean(np.abs((group["actual"] - group["pred"]) / (group["actual"] + 1e-8))) * 100),
            "relative_error": float(np.mean((group["pred"] - group["actual"]) / (group["actual"] + 1e-8))),
            "relative_error_abs": float(np.mean(np.abs((group["pred"] - group["actual"]) / (group["actual"] + 1e-8))))
        }

    group = df_eval[~mask_dia]
    if len(group) >= 2:
        metrics["night"] = {
            "r2": float(r2_score(group["actual"], group["pred"])),
            "mse": float(mean_squared_error(group["actual"], group["pred"])),
            "rmse": float(np.sqrt(mean_squared_error(group["actual"], group["pred"]))),
            "mae": float(mean_absolute_error(group["actual"], group["pred"])),
            "mape": float(np.mean(np.abs((group["actual"] - group["pred"]) / (group["actual"] + 1e-8))) * 100),
            "relative_error": float(np.mean((group["pred"] - group["actual"]) / (group["actual"] + 1e-8))),
            "relative_error_abs": float(np.mean(np.abs((group["pred"] - group["actual"]) / (group["actual"] + 1e-8))))
        }

    for h in range(1, horizon + 1):
        g_h = df_eval[df_eval["horizon"] == h]
        if len(g_h) < 2:
            continue
        metrics["per_horizon"][f"h_{h}"] = {
            "r2": float(r2_score(g_h["actual"], g_h["pred"])),
            "mse": float(mean_squared_error(g_h["actual"], g_h["pred"])),
            "rmse": float(np.sqrt(mean_squared_error(g_h["actual"], g_h["pred"]))),
            "mae": float(mean_absolute_error(g_h["actual"], g_h["pred"])),
            "mape": float(np.mean(np.abs((g_h["actual"] - g_h["pred"]) / (g_h["actual"] + 1e-8))) * 100)
        }

    for month, group in df_eval.groupby(pd.Grouper(freq="ME")):
        if len(group) < 2:
            continue
        metrics["per_month"][month.strftime("%Y-%m")] = {
            "r2": float(r2_score(group["actual"], group["pred"])),
            "mse": float(mean_squared_error(group["actual"], group["pred"])),
            "rmse": float(np.sqrt(mean_squared_error(group["actual"], group["pred"]))),
            "mae": float(mean_absolute_error(group["actual"], group["pred"])),
            "mape": float(np.mean(np.abs((group["actual"] - group["pred"]) / (group["actual"] + 1e-8))) * 100)
        }

    # ---------------------------
    # Save experiment (independent JSON per target variable)
    # ---------------------------
    network_params = {
        "batch_size": batch_size,
        "lr": lr,
        "num_epochs": num_epochs,
        "patience": patience,
        "window_size": window_size,
        "horizon": horizon,
        "kernel_size": kernel_size,
        "train_block_days": train_block_days,
        "val_block_days": val_block_days,
        "stride_train": stride_train,
        "stride_val": stride_val,
    }

    save_experiment_json(
        filepath=archivo_resultado,
        input_variables=input_columns,
        target_variable=output_column,
        model=model,
        optimizer=optimizer,
        network_params=network_params,
        training_time=training_time,
        metrics=metrics,
        predictions=prediction_path,
        notes="Univariate DLinear baseline: uses only the target variable's own history."
    )

    print(f"[{target_label}] Results saved to: {archivo_resultado}")


# ---------------------------
# Run one independent experiment per target variable
# ---------------------------
if __name__ == "__main__":
    for target in TARGETS:
        ejecutar_experimento(
            output_column=target["output_column"],
            input_column=target["input_column"],
            target_dir=target["target_dir"],
            target_label=target["name"],
        )
