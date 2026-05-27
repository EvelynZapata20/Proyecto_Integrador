"""Convierte credit_card_balance.csv a parquet en data/."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

CSV_PATH = DATA_DIR / "credit_card_balance.csv" / "credit_card_balance.csv"
PARQUET_PATH = DATA_DIR / "credit_card_balance.parquet"


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {CSV_PATH}")

    print(f"Leyendo {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"Filas: {len(df):,} | Columnas: {df.shape[1]}")

    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Guardado en {PARQUET_PATH}")


if __name__ == "__main__":
    main()
