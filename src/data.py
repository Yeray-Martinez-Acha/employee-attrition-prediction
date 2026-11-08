"""Carga de la configuración y de los datos, con validaciones básicas."""

from pathlib import Path

import pandas as pd
import yaml

# Raíz del proyecto: la carpeta que contiene src/
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    """Lee el fichero de configuración del proyecto."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_data(df: pd.DataFrame, config: dict) -> None:
    """Comprueba que los datos cumplen lo esperado antes de usarlos.

    Si algo falla, detiene la ejecución con un mensaje claro en lugar de
    dejar que el error aparezca más adelante en el modelo.
    """
    target = config["data"]["target"]
    errores = []

    columnas_necesarias = [target] + config["preprocessing"]["drop_columns"]
    faltan = [c for c in columnas_necesarias if c not in df.columns]
    if faltan:
        errores.append(f"Faltan columnas: {faltan}")

    if df.isna().any().any():
        errores.append("Hay valores nulos en el dataset")

    if df.duplicated().any():
        errores.append(f"Hay {df.duplicated().sum()} filas duplicadas")

    if target in df.columns and not set(df[target].unique()) <= {"Yes", "No"}:
        errores.append(f"{target} contiene valores distintos de 'Yes'/'No'")

    if errores:
        raise ValueError("Validación de datos fallida:\n- " + "\n- ".join(errores))


def load_raw_data(config: dict) -> pd.DataFrame:
    """Carga el CSV original y lo valida."""
    path = ROOT / config["data"]["raw_path"]
    df = pd.read_csv(path, encoding=config["data"]["encoding"])
    validate_data(df, config)
    return df