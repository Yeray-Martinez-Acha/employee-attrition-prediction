"""Evaluación del modelo: validación cruzada, registro en MLflow y elección del umbral."""

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from src.data import ROOT


def configurar_mlflow(config):
    """Indica a MLflow dónde guardar los experimentos y con qué nombre.

    Los resultados se registran en una base de datos SQLite (mlflow.db) y los
    archivos asociados, si los hay, en la carpeta mlartifacts/, ambos en la
    raíz del proyecto.
    """
    ruta_db = (ROOT / config["mlflow"]["tracking_uri"]).as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{ruta_db}")

    nombre = config["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(nombre) is None:
        mlflow.create_experiment(nombre, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(nombre)


def evaluar_variante(nombre, pipe, X, y, config, params=None):
    """Evalúa una variante con validación cruzada estratificada repetida y la registra en MLflow.

    Usa siempre las mismas particiones (semilla fija), de modo que todas las
    variantes se comparan en igualdad de condiciones. Devuelve las métricas
    medias y el AUC de cada partición, para poder comparar variantes por pares.
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
                                 random_state=config["project"]["random_state"])
    res = cross_validate(pipe, X, y, cv=cv,
                         scoring={"auc": "roc_auc", "brier": "neg_brier_score"})

    metricas = {
        "auc_media": res["test_auc"].mean(),
        "auc_std": res["test_auc"].std(),
        "brier_media": -res["test_brier"].mean(),  # sklearn lo devuelve en negativo
    }

    with mlflow.start_run(run_name=nombre):
        mlflow.log_params(params or {})
        mlflow.log_metrics(metricas)

    return {"variante": nombre, **metricas}, res["test_auc"]


def tabla_umbrales(y, prob, coste_perdida, coste_alarma, eficacia, umbrales):
    """Calcula, para cada umbral, la matriz de confusión, las métricas y el ahorro económico.

    - coste_perdida: coste de perder a cada empleado (un valor por empleado).
    - coste_alarma: coste de una conversación con un empleado señalado.
    - eficacia: proporción de las bajas detectadas que se consigue evitar.

    El ahorro se mide frente a no utilizar el modelo (perder a todos los que se van).
    """
    y = np.asarray(y)
    prob = np.asarray(prob)
    coste_perdida = np.asarray(coste_perdida)
    coste_sin_modelo = coste_perdida[y == 1].sum()

    filas = []
    for u in umbrales:
        alarma = prob >= u
        vp = int((alarma & (y == 1)).sum())
        fp = int((alarma & (y == 0)).sum())
        fn = int((~alarma & (y == 1)).sum())

        coste = (alarma.sum() * coste_alarma                                   # conversaciones
                 + coste_perdida[~alarma & (y == 1)].sum()                     # bajas no detectadas
                 + (1 - eficacia) * coste_perdida[alarma & (y == 1)].sum())    # detectadas no retenidas

        filas.append({
            "umbral": u,
            "alarmas": int(alarma.sum()),
            "pct_plantilla": alarma.mean() * 100,
            "VP": vp, "FP": fp, "FN": fn,
            "recall": vp / (vp + fn),
            "precision": vp / (vp + fp) if (vp + fp) > 0 else np.nan,
            "ahorro": coste_sin_modelo - coste,
        })
    return pd.DataFrame(filas)