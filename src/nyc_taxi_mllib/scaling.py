"""Benchmarks de escalabilidad del procesamiento distribuido.

Dos experimentos clasicos para evaluar arquitecturas distribuidas:

* **Escalabilidad de datos** (*data scaling*): se fija el paralelismo y se varia
  el volumen de entrenamiento; se mide el tiempo de ajuste del modelo.
* **Escalabilidad fuerte** (*strong scaling*): se fija el volumen y se varia el
  numero de cores; se mide el *speedup* respecto a 1 core.
"""
from __future__ import annotations

import time

from pyspark.ml import Estimator
from pyspark.sql import DataFrame

__all__ = ["time_fit", "data_scaling", "strong_scaling"]


def time_fit(estimator: Estimator, train: DataFrame) -> tuple[float, object]:
    """Entrena ``estimator`` sobre ``train`` y devuelve (segundos, modelo).

    Se fuerza la materializacion con ``count()`` antes de cronometrar para no
    medir tiempos de lectura perezosa.
    """
    train.count()  # materializa/cachea la entrada
    t0 = time.perf_counter()
    model = estimator.fit(train)
    elapsed = time.perf_counter() - t0
    return elapsed, model


def data_scaling(
    estimator: Estimator,
    train: DataFrame,
    fractions: list[float],
    seed: int = 42,
) -> list[dict[str, float]]:
    """Mide el tiempo de entrenamiento para fracciones crecientes de datos."""
    results = []
    for frac in fractions:
        sample = train if frac >= 1.0 else train.sample(False, frac, seed=seed)
        sample = sample.cache()
        n = sample.count()
        elapsed, _ = time_fit(estimator, sample)
        results.append({"fraction": frac, "n_rows": n, "train_seconds": elapsed})
        sample.unpersist()
    return results


def strong_scaling(
    build_estimator,
    build_train,
    core_counts: list[int],
) -> list[dict[str, float]]:
    """Mide *speedup* variando cores.

    ``build_estimator`` y ``build_train`` son fabricas que reciben una
    ``SparkSession`` (creada con un numero fijo de cores) y devuelven el
    estimador y el DataFrame de entrenamiento, respectivamente. Esto es
    necesario porque el numero de cores se fija al crear la sesion Spark.
    """
    from nyc_taxi_mllib.spark_session import build_spark

    results = []
    baseline = None
    for cores in core_counts:
        spark = build_spark(app_name=f"scaling-{cores}c", cores=cores)
        train = build_train(spark).cache()
        train.count()
        estimator = build_estimator(spark)
        elapsed, _ = time_fit(estimator, train)
        if baseline is None:
            baseline = elapsed
        results.append(
            {"cores": cores, "train_seconds": elapsed, "speedup": baseline / elapsed}
        )
        train.unpersist()
        spark.stop()
    return results
