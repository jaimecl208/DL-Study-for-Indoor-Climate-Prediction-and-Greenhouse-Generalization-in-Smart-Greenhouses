"""
Cartagena preprocessing (compartment 0).

This script starts from the raw (15 min) meteorology and Cartagena
compartment 0 data, and generates the final feature dataset ready for
modeling.

Pipeline:
    1. Load raw data (meteorology + compartment 0)
    2. Join both by timestamp index
    3. Filter out invalid days (maintenance, sensor outages, outliers, etc.)
    4. Interpolate gaps
    5. Generate temporal features (hour, day of week, month, day/night)
    6. Generate lagged variables (lags)
    7. Save the final dataset

Raw data is NOT included in this repository. See `data/README.md` for
its origin and download instructions. Adjust `RAW_METEO_CSV` and
`RAW_COMP_CSV` to the local path where you downloaded them.
"""

import os
import numpy as np
import pandas as pd
import ephem

# ======================
# CONFIG
# ======================

RAW_METEO_CSV = "./data/raw/Cartagena_meteo_15min.csv"  # not included in the repo, see data/README.md
RAW_COMP_CSV = "./data/raw/Cartagena_comp0_15min.csv"    # not included in the repo, see data/README.md
OUTPUT_CSV = "./data/processed/Cartagena_comp0_processed.csv"

DATE_COLUMN = "dateTime"

# NOTE: these coordinates match the Álava station.
# Check whether they are also correct for Cartagena before using this script.
LAT = 37.624934
LON = -0.997273

# Variables interpolated by time (no gap limit)
VARS_INTERPOLAR_TIME = [
    "WIND_DIRECTION_Cartagena_meteo",
    "DB_TEMP_EXT_Cartagena_meteo", "REL_HUM_EXT_Cartagena_meteo",
    "RADIATION_MEAN_Cartagena_meteo", "RADIATION_MIN_Cartagena_meteo",
    "RADIATION_MAX_Cartagena_meteo", "WIND_VELOCITY_MEAN_Cartagena_meteo",
    "WIND_VELOCITY_MIN_Cartagena_meteo", "WIND_VELOCITY_MAX_Cartagena_meteo",
    "VPD_Cartagena_comp0", "HUM_Cartagena_comp0",
    "REL_HUM_INT_Cartagena_comp0", "DB_TEMP_INT_Cartagena_comp0", "WB_TEMP_Cartagena_comp0",
]

# Variables interpolated by time, but only in internal, bounded gaps
# (limit=48 -> up to 12h, limit_area='inside' -> no extrapolation at the edges)
VARS_INTERPOLAR_TIME_HUECOS = ["WATER_DEF_Cartagena_comp0"]

# Categorical/discrete variables filled with the last valid value (forward fill)
VARS_INTERPOLAR_FFILL = [
    "DEST_Cartagena_comp0", "VENTIL_TEMP_Cartagena_comp0",
    "WINDOW_0_Cartagena_comp0", "WINDOW_1_Cartagena_comp0", "WINDOW_2_Cartagena_comp0",
]

# Days excluded from the dataset (maintenance, sensor outages, anomalous values...)
DIAS_A_ELIMINAR = [
    "2024-07-01", "2024-07-02", "2024-07-03", "2024-07-04", "2024-07-05",
    "2024-07-06", "2024-07-11", "2024-07-12", "2024-10-04",
    "2025-07-28",
    "2026-02-23", "2026-02-24",
]

# Variables for which lagged versions are generated
LAG_VARS = [
    "ACCUMULATED_RADIATION_MAX_Cartagena_meteo", "ACCUMULATED_RADIATION_MIN_Cartagena_meteo",
    "ACCUMULATED_RADIATION_MEAN_Cartagena_meteo", "DB_TEMP_INT_Cartagena_comp0",
    "REL_HUM_INT_Cartagena_comp0", "DB_TEMP_EXT_Cartagena_meteo", "REL_HUM_EXT_Cartagena_meteo",
    "WIND_VELOCITY_MEAN_Cartagena_meteo", "WIND_VELOCITY_MAX_Cartagena_meteo",
    "WIND_VELOCITY_MIN_Cartagena_meteo", "RADIATION_MEAN_Cartagena_meteo",
    "RADIATION_MIN_Cartagena_meteo", "RADIATION_MAX_Cartagena_meteo",
    "WIND_DIRECTION_Cartagena_meteo", "VPD_Cartagena_comp0", "HUM_Cartagena_comp0",
    "WB_TEMP_Cartagena_comp0", "WATER_DEF_Cartagena_comp0",
]

# Lags to apply (1 step = 15 min)
LAGS = [2, 4]


# ======================
# LOADING
# ======================

def cargar_y_preparar_df(ruta_archivo: str) -> pd.DataFrame:
    """Load a CSV, parse the timestamp column and set it as the index."""
    df = pd.read_csv(ruta_archivo)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.set_index(DATE_COLUMN)
    return df


# ======================
# CLEANING
# ======================

def filtrar_dataset_por_dias(df: pd.DataFrame, dias_a_eliminar: list[str]) -> pd.DataFrame:
    """Remove the days listed in `dias_a_eliminar` from the dataset."""
    dias_dt = pd.to_datetime(dias_a_eliminar)
    return df[~df.index.normalize().isin(dias_dt)]


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
    # --- Load ---
    df_meteo = cargar_y_preparar_df(RAW_METEO_CSV)
    df_comp0 = cargar_y_preparar_df(RAW_COMP_CSV)

    # Left join to preserve the compartment's timestamps
    df = df_comp0.join(df_meteo, how="left")

    # --- Cleaning ---
    df = filtrar_dataset_por_dias(df, DIAS_A_ELIMINAR)
    df = interpolar_dataset(df)

    # --- Prepare for feature engineering ---
    df = df.fillna(-1)

    # --- Features ---
    df = anadir_features_temporales(df)
    df = anadir_lags(df, LAG_VARS, LAGS)
    df = df.dropna()

    # Target variables duplicated as inputs, to avoid errors during modeling
    df["DB_TEMP_INT_INPUT_Cartagena_comp0"] = df["DB_TEMP_INT_Cartagena_comp0"]
    df["REL_HUM_INT_INPUT_Cartagena_comp0"] = df["REL_HUM_INT_Cartagena_comp0"]

    # --- Save ---
    print(df.head())
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV)
    print(f"File saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
