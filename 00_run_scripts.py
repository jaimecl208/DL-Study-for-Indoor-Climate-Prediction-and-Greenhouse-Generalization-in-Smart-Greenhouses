"""
Runs every experiment script the configured number of times.

The number of runs per architecture is set in REPEATS below (models not
listed fall back to DEFAULT_REPEATS). An entry in SCRIPTS can override it
with an optional third element.

Run from the repo root:
    python 00_run_scripts.py
"""

import subprocess
import sys
import time
import os
from pathlib import Path

# ---------------------------
# Config
# ---------------------------

PYTHON_PATH = os.path.join(".venv", "Scripts", "python.exe")

# How many times each architecture's scripts should run. Falls back to
# DEFAULT_REPEATS for any model not listed here.
DEFAULT_REPEATS = 5
REPEATS = {
    "lstm": 5,
    "gru": 5,
    "cnn_lstm": 5,
    "dlinear": 5,
    "nlinear": 5,
    "patchtst": 5,
    "itransformer": 5,
    "finetuning": 3,
}

SCRIPTS = [
    #(model, script_path, nº runs - optional )
    ("lstm", "scripts/02_experiments/alava/01_lstm_alava_temp.py"),
    ("lstm", "scripts/02_experiments/alava/02_lstm_alava_hum.py"),
    ("gru", "scripts/02_experiments/alava/03_gru_alava_temp.py"),
    ("gru", "scripts/02_experiments/alava/04_gru_alava_hum.py"),
    ("cnn_lstm", "scripts/02_experiments/alava/05_cnn_lstm_alava_temp.py"),
    ("cnn_lstm", "scripts/02_experiments/alava/06_cnn_lstm_alava_hum.py"),
    ("dlinear", "scripts/02_experiments/alava/07_dlinear_alava.py"),
    ("nlinear", "scripts/02_experiments/alava/08_nlinear_alava.py"),
    ("patchtst", "scripts/02_experiments/alava/09_patchtst_alava.py"),
    ("itransformer", "scripts/02_experiments/alava/10_itransformer_alava.py"),

    ("lstm", "scripts/02_experiments/cartagena/01_lstm_cartagena_temp.py"),
    ("lstm", "scripts/02_experiments/cartagena/02_lstm_cartagena_hum.py"),
    ("gru", "scripts/02_experiments/cartagena/03_gru_cartagena_temp.py"),
    ("gru", "scripts/02_experiments/cartagena/04_gru_cartagena_hum.py"),
    ("cnn_lstm", "scripts/02_experiments/cartagena/05_cnn_lstm_cartagena_temp.py"),
    ("cnn_lstm", "scripts/02_experiments/cartagena/06_cnn_lstm_cartagena_hum.py"),
    ("dlinear", "scripts/02_experiments/cartagena/07_dlinear_cartagena.py"),
    ("nlinear", "scripts/02_experiments/cartagena/08_nlinear_cartagena.py"),
    ("patchtst", "scripts/02_experiments/cartagena/09_patchtst_cartagena.py"),
    ("itransformer", "scripts/02_experiments/cartagena/10_itransformer_cartagena.py"),

    ("finetuning", "scripts/02_experiments/finetuning/01_itransformer_alava_cartagena_finetuning_curves.py", 1),
    ("finetuning", "scripts/02_experiments/finetuning/02_itransformer_alava_cartagena_direct_comparison.py"),
    ("finetuning", "scripts/02_experiments/finetuning/03_itransformer_cartagena_alava_finetuning_curves.py", 1),
    ("finetuning", "scripts/02_experiments/finetuning/04_itransformer_cartagena_alava_direct_comparison.py"),
]

LOG_DIR = Path("results/logs")


def run_script(script_path: str, run_number: int, total_runs: int) -> bool:
    """Runs one script once and returns True on success."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    script_name = Path(script_path).stem
    log_file = LOG_DIR / f"{script_name}_run{run_number}.log"

    print(f"\n{'=' * 70}")
    print(f"[{run_number}/{total_runs}] Running: {script_path}")
    print(f"{'=' * 70}")

    start = time.time()
    with open(log_file, "w") as f:
        result = subprocess.run(
            [PYTHON_PATH, script_path],
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - start

    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit code {result.returncode})"
    print(f"-> {status} in {elapsed:.1f}s. Log: {log_file}")
    return ok


def resolve_repeats(entry: tuple) -> tuple[str, str, int]:
    """Devuelve (model, script_path, repeats).

    Si la tupla trae un tercer elemento, se usa como número de repeticiones;
    si no, se cae a REPEATS[model] o DEFAULT_REPEATS.
    """
    model, script_path, *rest = entry

    if rest:
        repeats = rest[0]
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
            raise ValueError(
                f"Número de repeticiones inválido para {script_path}: {repeats!r}"
            )
    else:
        repeats = REPEATS.get(model, DEFAULT_REPEATS)

    return model, script_path, repeats


def main() -> None:
    summary = []

    for entry in SCRIPTS:
        model, script_path, repeats = resolve_repeats(entry)
        for run_number in range(1, repeats + 1):
            ok = run_script(script_path, run_number, repeats)
            summary.append((script_path, run_number, ok))

    # ---------------------------
    # Final report
    # ---------------------------
    print(f"\n{'#' * 70}")
    print("SUMMARY")
    print(f"{'#' * 70}")

    failures = [s for s in summary if not s[2]]
    print(f"Total runs: {len(summary)} | Failed: {len(failures)}")

    if failures:
        print("\nFailed runs:")
        for script_path, run_number, _ in failures:
            print(f"  - {script_path} (run {run_number})")
        sys.exit(1)

    print("\nAll runs completed successfully.")


if __name__ == "__main__":
    main()