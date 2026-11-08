"""Construcción de nuevas variables mediante transformadores de scikit-learn.

Cada transformador aprende lo que necesita en fit (solo con el conjunto de
entrenamiento) y lo aplica en transform, lo que evita la fuga de datos.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DIMENSIONES_SATISFACCION = (
    "EnvironmentSatisfaction",
    "JobSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
)


class SalarioRelativo(BaseEstimator, TransformerMixin):
    """Sustituye el salario por el salario relativo a la mediana de su nivel.

    Corrige la colinealidad entre MonthlyIncome y JobLevel: conserva la
    información de si el empleado cobra más o menos que sus compañeros de
    nivel sin duplicar la información del propio nivel.
    """

    def __init__(self, col_salario="MonthlyIncome", col_nivel="JobLevel"):
        self.col_salario = col_salario
        self.col_nivel = col_nivel

    def fit(self, X, y=None):
        # Medianas aprendidas solo con los datos de entrenamiento
        self.medianas_ = X.groupby(self.col_nivel)[self.col_salario].median()
        self.mediana_global_ = X[self.col_salario].median()
        return self

    def transform(self, X):
        X = X.copy()
        # Si apareciera un nivel no visto en el entrenamiento, se usa la mediana global
        mediana_nivel = X[self.col_nivel].map(self.medianas_).fillna(self.mediana_global_)
        X["SalarioRelativo"] = X[self.col_salario] / mediana_nivel
        return X.drop(columns=self.col_salario)


class IndicadorInsatisfaccion(BaseEstimator, TransformerMixin):
    """Cuenta en cuántas dimensiones de satisfacción el empleado está en el nivel 1.

    El análisis exploratorio mostró que el aumento de la rotación se concentra
    en el nivel más bajo de cada escala y que las escalas no están
    correlacionadas entre sí, por lo que un recuento refleja mejor el riesgo
    que una media.
    """

    def __init__(self, columnas=DIMENSIONES_SATISFACCION, eliminar_originales=False):
        self.columnas = columnas
        self.eliminar_originales = eliminar_originales

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["NumInsatisfaccion"] = (X[list(self.columnas)] == 1).sum(axis=1)
        if self.eliminar_originales:
            X = X.drop(columns=list(self.columnas))
        return X


class Estancamiento(BaseEstimator, TransformerMixin):
    """Años sin ascenso en relación con la antigüedad en la empresa.

    Se suma 1 a la antigüedad para evitar divisiones entre cero en los
    empleados con menos de un año en la empresa.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Estancamiento"] = X["YearsSinceLastPromotion"] / (X["YearsAtCompany"] + 1)
        return X


class TieneStockOptions(BaseEstimator, TransformerMixin):
    """Sustituye StockOptionLevel por un indicador binario (1 = tiene stock options).

    El análisis exploratorio mostró que el efecto se concentra en no tener
    opciones (nivel 0), sin una relación ordenada entre los niveles 1 a 3.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TieneStockOptions"] = (X["StockOptionLevel"] > 0).astype(int)
        return X.drop(columns="StockOptionLevel")