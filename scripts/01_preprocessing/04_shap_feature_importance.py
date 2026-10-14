"""
SHAP feature-importance analysis for the Álava and Cartagena (compartment
0) processed datasets.

For each location and each target (indoor temperature and relative
humidity) a RandomForestRegressor is trained on the first 70% of the
series (chronological split) and SHAP values are computed with
TreeExplainer on a sample of the remaining 30%.

Set REMOVE_PROXY to True to drop the climate actuators (windows, shade
screen), which react to the indoor climate and act as proxies of the
target.

The `*_INPUT_*` columns created during preprocessing are exact copies of
the targets, so they are always dropped to avoid target leakage. Lagged
features (`*_lag*`) are also excluded.

One SHAP summary (beeswarm) plot is saved per target, under
figures/<location>/shap/.

Run from the repo root:
    python scripts/01_preprocessing/04_shap_feature_importance.py
"""

import os
import time

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: avoids Tk errors with n_jobs=-1

import matplotlib.pyplot as plt
import pandas as pd
import shap

from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# Config
# ---------------------------

COLUMNA_FECHA = "dateTime"

LOCATIONS = {
    "alava": {
        "csv_path": "./data/processed/Alava_processed.csv",
        "output_dir": "./figures/alava/shap",
        "targets": {
            "temperature": "DB_TEMP_INT_Alava",
            "rel_humidity": "REL_HUM_INT_Alava",
        },
        "proxy_features": [
            "WINDOW_0_Alava",
            "WINDOW_1_Alava",
            "SHADE_SCREEN_Alava",
        ],
    },
    "cartagena": {
        "csv_path": "./data/processed/Cartagena_comp0_processed.csv",
        "output_dir": "./figures/cartagena/shap",
        "targets": {
            "temperature": "DB_TEMP_INT_Cartagena_comp0",
            "rel_humidity": "REL_HUM_INT_Cartagena_comp0",
        },
        "proxy_features": [
            "WINDOW_0_Cartagena_comp0",
            "WINDOW_1_Cartagena_comp0",
            "WINDOW_2_Cartagena_comp0",
        ],
    },
}

TRAIN_RATIO = 0.7
SAMPLE_SIZE = 5000
RANDOM_STATE = 42
REMOVE_PROXY = True


# ---------------------------
# Functions
# ---------------------------

def cargar_datos(csv_path: str) -> pd.DataFrame:
    """Load a processed CSV, sort it chronologically and drop target copies."""
    df = pd.read_csv(csv_path, parse_dates=[COLUMNA_FECHA])
    df = df.fillna(-1)
    df = df.sort_values(COLUMNA_FECHA)
    df = df.drop(columns=[COLUMNA_FECHA])

    # *_INPUT_* columns are exact copies of the targets (target leakage)
    # and lagged features are excluded from the analysis
    columnas_excluidas = [
        c for c in df.columns
        if "_INPUT_" in c or "_lag" in c
    ]
    return df.drop(columns=columnas_excluidas)


def run_experiment(
    df: pd.DataFrame,
    location: str,
    target_name: str,
    target: str,
    proxy_features: list[str],
    output_dir: str,
    remove_proxy: bool,
) -> None:
    experiment_name = "without_proxy_features" if remove_proxy else "all_features"

    print(f"\n{'=' * 60}")
    print(f"{location} | {target} | {experiment_name}")
    print(f"{'=' * 60}")

    current_df = df.copy()

    if remove_proxy:
        cols_to_remove = [
            c for c in proxy_features
            if c != target and c in current_df.columns
        ]
        current_df = current_df.drop(columns=cols_to_remove)

    # --- Chronological split ---
    split_index = int(len(current_df) * TRAIN_RATIO)

    train = current_df.iloc[:split_index]
    test = current_df.iloc[split_index:]

    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])

    print(f"Train: {X_train.shape}")
    print(f"Test: {X_test.shape}")

    # --- Model ---
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # --- SHAP sample ---
    if SAMPLE_SIZE is not None and len(X_test) > SAMPLE_SIZE:
        X_shap = X_test.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
    else:
        X_shap = X_test

    print(f"SHAP samples: {len(X_shap)}")

    # --- SHAP ---
    print("Computing SHAP values...")
    start = time.time()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap, approximate=True)

    print(f"Time: {time.time() - start:.2f} s")

    # --- Summary plot ---
    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)

    figure_path = os.path.join(
        output_dir,
        f"{location.capitalize()}_{target_name}_{experiment_name}_shap_summary.png",
    )
    plt.savefig(figure_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved: {figure_path}")


# ---------------------------
# Pipeline
# ---------------------------

def main() -> None:
    for location, cfg in LOCATIONS.items():
        print(f"\n########## LOCATION: {location} ##########")

        df = cargar_datos(cfg["csv_path"])
        os.makedirs(cfg["output_dir"], exist_ok=True)

        for target_name, target in cfg["targets"].items():
            run_experiment(
                df=df,
                location=location,
                target_name=target_name,
                target=target,
                proxy_features=cfg["proxy_features"],
                output_dir=cfg["output_dir"],
                remove_proxy=REMOVE_PROXY,
            )

    print("\nAll SHAP experiments finished.")


if __name__ == "__main__":
    main()
