"""
Temporally aligns the Álava and Cartagena (compartment 0) feature
datasets, and differentiates by location the name of any generic
variable (temporal features such as 'hour', 'month', 'is_day'...) that
is not already differentiated, so they can be used without ambiguity in
finetuning experiments.

IMPORTANT: gaps are not filled in. Both datasets already have missing
timestamps removed on purpose (real gaps, not errors). This script
respects those gaps: it computes the intersection of timestamps between
the two CSVs and filters each one to that intersection, without
reindexing or interpolating anything.

Output: two files, one per location, with the exact same dates and with
every column unambiguously identified (e.g. 'hour' becomes 'hour_Alava'
/ 'hour_Cartagena_comp0').
"""

import os

import pandas as pd

# ---------------------------
# Config: input/output paths
# ---------------------------
PATH_ALAVA = "./data/processed/Alava_processed.csv"
PATH_CARTAGENA = "./data/processed/Cartagena_comp0_processed.csv"

PATH_SALIDA_ALAVA = "./data/processed/Alava_processed_aligned.csv"
PATH_SALIDA_CARTAGENA = "./data/processed/Cartagena_comp0_processed_aligned.csv"

COLUMNA_FECHA = "dateTime"

# Location marker already present in each dataset's own columns. Any
# column that does NOT contain this text is considered generic and gets
# the corresponding suffix appended.
MARCADOR_ALAVA = "Alava"
SUFIJO_ALAVA = "_Alava"

MARCADOR_CARTAGENA = "Cartagena"
SUFIJO_CARTAGENA = "_Cartagena_comp0"


# ---------------------------
# Functions
# ---------------------------

def diferenciar_columnas_genericas(df: pd.DataFrame, marcador: str, sufijo: str) -> pd.DataFrame:
    """Append `sufijo` to columns that do not already contain `marcador` in their name."""
    renombres = {
        col: f"{col}{sufijo}"
        for col in df.columns
        if marcador not in col
    }
    return df.rename(columns=renombres)


def alinear_por_timestamps_comunes(
    df1: pd.DataFrame, df2: pd.DataFrame, columna_fecha: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter both DataFrames to the intersection of timestamps, without filling gaps."""
    timestamps_comunes = set(df1[columna_fecha]) & set(df2[columna_fecha])

    df1_alineado = (
        df1[df1[columna_fecha].isin(timestamps_comunes)]
        .sort_values(columna_fecha)
        .reset_index(drop=True)
    )
    df2_alineado = (
        df2[df2[columna_fecha].isin(timestamps_comunes)]
        .sort_values(columna_fecha)
        .reset_index(drop=True)
    )

    assert list(df1_alineado[columna_fecha]) == list(df2_alineado[columna_fecha]), \
        "Timestamps of both aligned CSVs do not match exactly."

    return df1_alineado, df2_alineado


# ---------------------------
# Pipeline
# ---------------------------

def main() -> None:
    # --- Load ---
    df_alava = pd.read_csv(PATH_ALAVA, parse_dates=[COLUMNA_FECHA])
    df_cartagena = pd.read_csv(PATH_CARTAGENA, parse_dates=[COLUMNA_FECHA])

    print(f"Álava original: {len(df_alava)} rows "
          f"({df_alava[COLUMNA_FECHA].min()} -> {df_alava[COLUMNA_FECHA].max()})")
    print(f"Cartagena comp0 original: {len(df_cartagena)} rows "
          f"({df_cartagena[COLUMNA_FECHA].min()} -> {df_cartagena[COLUMNA_FECHA].max()})")

    # --- Differentiate generic variables ---
    df_alava = diferenciar_columnas_genericas(df_alava, MARCADOR_ALAVA, SUFIJO_ALAVA)
    df_cartagena = diferenciar_columnas_genericas(df_cartagena, MARCADOR_CARTAGENA, SUFIJO_CARTAGENA)

    # --- Temporal alignment (timestamp intersection, no gap filling) ---
    df_alava_alineado, df_cartagena_alineado = alinear_por_timestamps_comunes(
        df_alava, df_cartagena, COLUMNA_FECHA
    )

    print(f"\nCommon timestamps: {len(df_alava_alineado)}")

    # --- Save ---
    os.makedirs(os.path.dirname(PATH_SALIDA_ALAVA), exist_ok=True)
    os.makedirs(os.path.dirname(PATH_SALIDA_CARTAGENA), exist_ok=True)
    df_alava_alineado.to_csv(PATH_SALIDA_ALAVA, index=False)
    df_cartagena_alineado.to_csv(PATH_SALIDA_CARTAGENA, index=False)

    print(f"\nSaved: {PATH_SALIDA_ALAVA}")
    print(f"Saved: {PATH_SALIDA_CARTAGENA}")


if __name__ == "__main__":
    main()
