"""Evaluacion de modelos y manejo de desbalance de clases.

Reproduce el enfoque de evaluacion de clase: ``BinaryClassificationEvaluator``
(AUC) y ``MulticlassClassificationEvaluator`` (accuracy, precision, recall, F1),
reportando ademas el *recall de la clase minoritaria* (propinas altas), que es
la metrica honesta bajo fuerte desbalance.
"""
from __future__ import annotations

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

__all__ = ["add_class_weights", "class_balance", "evaluate_predictions"]


def class_balance(df: DataFrame, label_col: str = "label") -> dict[str, float]:
    """Devuelve el conteo y la proporcion de cada clase."""
    counts = {int(r[label_col]): int(r["c"]) for r in df.groupBy(label_col).agg(F.count("*").alias("c")).collect()}
    n0, n1 = counts.get(0, 0), counts.get(1, 0)
    total = n0 + n1
    return {
        "n_neg": n0,
        "n_pos": n1,
        "total": total,
        "pos_rate": n1 / total if total else 0.0,
        "imbalance_ratio": (n0 / n1) if n1 else float("inf"),
    }


def add_class_weights(df: DataFrame, label_col: str = "label") -> DataFrame:
    """Agrega ``class_weight`` inversamente proporcional a la frecuencia.

    Cada clase recibe peso ``total / (2 * n_clase)``, de modo que ambas clases
    contribuyen por igual a la perdida pese al desbalance.
    """
    bal = class_balance(df, label_col)
    total, n0, n1 = bal["total"], bal["n_neg"], bal["n_pos"]
    w0 = total / (2.0 * n0) if n0 else 0.0
    w1 = total / (2.0 * n1) if n1 else 0.0
    return df.withColumn(
        "class_weight",
        F.when(F.col(label_col) == 1, F.lit(w1)).otherwise(F.lit(w0)),
    )


def evaluate_predictions(
    predictions: DataFrame, label_col: str = "label"
) -> dict[str, float]:
    """Calcula AUC, accuracy, precision/recall/F1 (ponderados) y recall+ minoria."""
    auc = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(predictions)

    def _mc(metric: str) -> float:
        return MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol="prediction", metricName=metric
        ).evaluate(predictions)

    # Recall de la clase positiva (propina alta) calculado explicitamente.
    tp = predictions.where((F.col(label_col) == 1) & (F.col("prediction") == 1)).count()
    fn = predictions.where((F.col(label_col) == 1) & (F.col("prediction") == 0)).count()
    recall_pos = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "auc": auc,
        "accuracy": _mc("accuracy"),
        "f1": _mc("f1"),
        "precision_w": _mc("weightedPrecision"),
        "recall_w": _mc("weightedRecall"),
        "recall_pos": recall_pos,
    }
