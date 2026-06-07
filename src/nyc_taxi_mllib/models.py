"""Catalogo de clasificadores distribuidos de Spark MLlib a comparar.

Todos los algoritmos provienen del temario del curso (Clase 5, *Scaling Out
Machine Learning con Spark*): regresion logistica como linea base, y arboles
de decision, random forest y gradient-boosted trees como modelos no lineales.
"""
from __future__ import annotations

from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)

__all__ = ["make_models"]


def make_models(
    features_col: str = "features",
    label_col: str = "label",
    weight_col: str | None = "class_weight",
):
    """Devuelve un diccionario ``{nombre: estimador}`` listo para entrenar.

    Si ``weight_col`` no es ``None``, los modelos que soportan ponderacion de
    instancias la usan para mitigar el desbalance de clases.
    """
    lr = LogisticRegression(
        featuresCol=features_col, labelCol=label_col, maxIter=50, regParam=0.0
    )
    dt = DecisionTreeClassifier(
        featuresCol=features_col, labelCol=label_col, maxDepth=10, maxBins=64
    )
    rf = RandomForestClassifier(
        featuresCol=features_col, labelCol=label_col, numTrees=50, maxDepth=10, maxBins=64
    )
    gbt = GBTClassifier(
        featuresCol=features_col, labelCol=label_col, maxIter=50, maxDepth=6, maxBins=64
    )

    if weight_col is not None:
        # Los cuatro clasificadores soportan ponderacion de instancias en
        # Spark 3.x (GBTClassifier la admite desde Spark 3.0, SPARK-19591),
        # lo que garantiza una comparacion equitativa frente al desbalance.
        lr = lr.setWeightCol(weight_col)
        rf = rf.setWeightCol(weight_col)
        dt = dt.setWeightCol(weight_col)
        gbt = gbt.setWeightCol(weight_col)

    return {
        "LogisticRegression": lr,
        "DecisionTree": dt,
        "RandomForest": rf,
        "GBT": gbt,
    }
