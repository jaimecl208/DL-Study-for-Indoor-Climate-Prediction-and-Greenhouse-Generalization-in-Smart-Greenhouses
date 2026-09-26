"""
Alava preprocessing.

This script starts from the raw (15 min) Alava station data and
generates the final feature dataset ready for modeling.

Pipeline:
    1. Load raw data
    2. Filter out invalid days (maintenance, sensor outages, outliers, etc.)
    3. Interpolate gaps
    4. Generate temporal features (hour, day of week, month, day/night)
    5. Generate lagged variables (lags)
    6. Save the final dataset

Raw data is NOT included in this repository. See `data/README.md` for
its origin and download instructions. Adjust `RAW_INPUT_CSV` to the
local path where you downloaded it.
"""

import os

import numpy as np
import pandas as pd
import ephem

# ======================
# CONFIG
# ======================

RAW_INPUT_CSV = "./data/raw/Alava_15min.csv"       # not included in the repo, see data/README.md
OUTPUT_CSV = "./data/processed/Alava_processed.csv"

DATE_COLUMN = "dateTime"

# Station coordinates, used to compute day/night
LAT = 42.8526
LON = -2.62629

# Variables interpolated by time (no gap limit)
VARS_INTERPOLAR_TIME = [
    "ACCUMULATED_RADIATION_MAX_Alava", "ACCUMULATED_RADIATION_MIN_Alava",
    "ACCUMULATED_RADIATION_MEAN_Alava", "DB_TEMP_INT_Alava", "REL_HUM_INT_Alava",
    "DB_TEMP_EXT_Alava", "REL_HUM_EXT_Alava", "WIND_VELOCITY_MEAN_Alava",
    "WIND_VELOCITY_MAX_Alava", "WIND_VELOCITY_MIN_Alava", "RADIATION_MEAN_Alava",
    "RADIATION_MIN_Alava", "RADIATION_MAX_Alava", "WIND_DIRECTION_Alava",
]

# Variables interpolated by time, but only in internal, bounded gaps
# (limit=48 -> up to 12h, limit_area='inside' -> no extrapolation at the edges)
VARS_INTERPOLAR_TIME_HUECOS = [
    "FLUID_TEMP_MEAN_Alava", "FLUID_TEMP_MIN_Alava", "FLUID_TEMP_MAX_Alava",
    "PIPE_TEMP_Alava",
]

# Categorical/discrete variables filled with the last valid value (forward fill)
VARS_INTERPOLAR_FFILL = [
    "SHADE_SCREEN_Alava", "WINDOW_0_Alava", "WINDOW_1_Alava",
    "VENTIL_TEMP_Alava", "ACT_FORCED_VENT_STAGE_Alava",
]

# Days excluded from the dataset (maintenance, sensor outages, anomalous values...)
DIAS_A_ELIMINAR = [
    "2023-12-24", "2023-12-25", "2023-12-26",
    "2025-07-28",
    "2025-08-16", "2025-08-17",
    "2025-08-26", "2025-08-27", "2025-08-28", "2025-08-29", "2025-08-30",
    "2025-08-31", "2025-09-01", "2025-09-02", "2025-09-03", "2025-09-04",
    "2025-09-05", "2025-09-06", "2025-09-07", "2025-09-08", "2025-09-09",
    "2026-02-23", "2026-02-24",
]

# Variables for which lagged versions are generated
LAG_VARS = VARS_INTERPOLAR_TIME + VARS_INTERPOLAR_TIME_HUECOS

# Lags to apply (1 step = 15 min)
LAGS = [2, 4]


# ======================
# CLEANING
# ======================

def filtrar_dataset_por_dias(df: pd.DataFrame, dias_a_eliminar: list[str]) -> pd.DataFrame:
    """Remove the days listed in `dias_a_eliminar` from the dataset."""
    return df[~df.index.strftime("%Y-%m-%d").isin(dias_a_eliminar)]


def interpolar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate/fill dataset gaps depending on each variable's type."""
    columnas = VARS_INTERPOLAR_FFILL + VARS_INTERPOLAR_TIME + VARS_INTERPOLAR_TIME_HUECOS
    df = df[columnas].copy()

    df[VARS_INTERPOLAR_TIME] = df[VARS_INTERPOLAR_TIME].interpolate(method="time")
    df[VARS_INTERPOLAR_TIME_HUECOS] = df[VARS_INTERPOLAR_TIME_HUECOS].interpolate(
        method="time", limit=48, limit_area="inside"
    )
    df[VARS_INTERPOLAR_FFILL] = df[VARS_INTERPOLAR_FFILL].ffill()

    return df


# ======================
# FEATURE ENGINEERING
# ======================

def is_it_day(date: pd.Timestamp, lat: float, lon: float) -> bool:
    """Return whether `date` falls during daytime at location (lat, lon)."""
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.date = date
    sun = ephem.Sun(observer)

    return observer.previous_rising(sun) > observer.previous_setting(sun)


def anadir_features_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day of week, month (cyclically encoded) and a day/night flag."""
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    fechas_unicas = pd.Series(df.index.unique())
    es_de_dia = fechas_unicas.apply(lambda dt: is_it_day(dt, LAT, LON)).astype(int)
    dia_dict = dict(zip(fechas_unicas, es_de_dia))
    df["is_day"] = df.index.map(dia_dict)

    return df


def anadir_lags(df: pd.DataFrame, variables: list[str], lags: list[int]) -> pd.DataFrame:
    """Add lagged columns for the given variables."""
    for col in variables:
        if col not in df.columns:
            print(f"WARNING: {col} does not exist, skipping")
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


# ======================
# PIPELINE
# ======================

def main() -> None:
    # --- Load and clean ---
    df = pd.read_csv(RAW_INPUT_CSV)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.set_index(DATE_COLUMN)

    df = filtrar_dataset_por_dias(df, DIAS_A_ELIMINAR)
    df = interpolar_dataset(df)

    # --- Prepare for feature engineering ---
    df = df.fillna(-1)

    # --- Features ---
    df = anadir_features_temporales(df)
    df = anadir_lags(df, LAG_VARS, LAGS)
    df = df.dropna()

    # Target variables duplicated as inputs, to avoid errors during modeling
    df["DB_TEMP_INT_INPUT_Alava"] = df["DB_TEMP_INT_Alava"]
    df["REL_HUM_INT_INPUT_Alava"] = df["REL_HUM_INT_Alava"]

    # --- Save ---
    print(df.head())
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV)
    print(f"File saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
