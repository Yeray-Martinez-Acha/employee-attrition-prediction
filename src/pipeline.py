"""Construcción del pipeline completo: nuevas variables, codificación, escalado y modelo."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

from src.features import (DIMENSIONES_SATISFACCION, IndicadorInsatisfaccion,
                          SalarioRelativo, TieneStockOptions)


def construir_pipeline(config, nuevas_variables=True, solo_indice=False,
                       excluir_sensibles=False, log_asimetricas=False):
    """Devuelve el pipeline completo (preprocesado + regresión logística).

    Los argumentos permiten construir las variantes que se comparan:
    - nuevas_variables: usa las variables de src/features.py (True) o las
      originales MonthlyIncome y StockOptionLevel (False).
    - solo_indice: sustituye las cuatro escalas de satisfacción por el recuento.
    - excluir_sensibles: elimina Gender, MaritalStatus y Age.
    - log_asimetricas: aplica log(1 + x) a las variables más asimétricas.
    """
    f = config["features"]
    numericas = list(f["numericas"])
    ordinales = list(f["ordinales"])
    binarias = list(f["binarias"])
    nominales = dict(f["nominales"])

    # 1. Construcción de nuevas variables (o variables originales equivalentes)
    if nuevas_variables:
        pasos = [
            ("salario_relativo", SalarioRelativo()),
            ("insatisfaccion", IndicadorInsatisfaccion()),
            ("stock_options", TieneStockOptions()),
        ]
    else:
        pasos = []
        numericas = [c for c in numericas
                     if c not in ("SalarioRelativo", "NumInsatisfaccion")] + ["MonthlyIncome"]
        binarias.remove("TieneStockOptions")
        ordinales.append("StockOptionLevel")

    # 2. Variables que se excluyen según la variante
    excluir = set()
    if solo_indice and nuevas_variables:
        excluir |= set(DIMENSIONES_SATISFACCION)
    if excluir_sensibles:
        excluir |= set(f["sensibles"])

    numericas = [c for c in numericas if c not in excluir]
    ordinales = [c for c in ordinales if c not in excluir]
    binarias = [c for c in binarias if c not in excluir]
    nominales = {c: ref for c, ref in nominales.items() if c not in excluir}

    asimetricas = [c for c in f["asimetricas"] if c in numericas] if log_asimetricas else []
    escaladas = [c for c in numericas + ordinales if c not in asimetricas]

    # 3. Tratamiento de cada tipo de variable
    transformadores = [
        ("num", StandardScaler(), escaladas),
        ("viajes", make_pipeline(OrdinalEncoder(categories=[f["viajes_orden"]]),
                                 StandardScaler()), ["BusinessTravel"]),
        ("bin", OneHotEncoder(drop="if_binary", sparse_output=False), binarias),
        ("nom", OneHotEncoder(drop=list(nominales.values()), sparse_output=False),
         list(nominales)),
    ]
    if asimetricas:
        transformadores.append(
            ("log", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                  StandardScaler()), asimetricas))

    preprocesado = ColumnTransformer(transformadores, remainder="drop",
                                     verbose_feature_names_out=False)
    preprocesado.set_output(transform="pandas")

    # 4. Modelo: regresión logística sin reponderar clases (probabilidades calibradas)
    modelo = LogisticRegression(max_iter=config["model"]["max_iter"])

    return Pipeline(pasos + [("preprocesado", preprocesado), ("modelo", modelo)])