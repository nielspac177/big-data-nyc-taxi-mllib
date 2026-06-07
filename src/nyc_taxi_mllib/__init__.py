"""nyc_taxi_mllib -- Comparacion de clasificadores distribuidos (Spark MLlib).

Prediccion de *propina alta* en viajes de taxi de NYC sobre decenas de millones
de registros, con enfasis en el manejo de desbalance de clases y la
escalabilidad del procesamiento distribuido.

API publica:
    build_spark            -- fabrica de SparkSession local
    load_trips             -- ingesta de los Parquet de NYC TLC
    clean_trips, add_label, build_feature_pipeline  -- preparacion de datos
    make_models            -- catalogo de clasificadores MLlib
    add_class_weights, class_balance, evaluate_predictions  -- evaluacion
    data_scaling, strong_scaling, time_fit          -- benchmarks
"""
from __future__ import annotations

from nyc_taxi_mllib.evaluate import (
    add_class_weights,
    class_balance,
    evaluate_predictions,
)
from nyc_taxi_mllib.features import (
    LOW_TIP_THRESHOLD,
    add_label,
    build_feature_pipeline,
    clean_trips,
)
from nyc_taxi_mllib.ingest import load_trips
from nyc_taxi_mllib.models import make_models
from nyc_taxi_mllib.scaling import data_scaling, strong_scaling, time_fit
from nyc_taxi_mllib.spark_session import build_spark

__all__ = [
    "build_spark",
    "load_trips",
    "clean_trips",
    "add_label",
    "build_feature_pipeline",
    "HIGH_TIP_THRESHOLD",
    "make_models",
    "add_class_weights",
    "class_balance",
    "evaluate_predictions",
    "data_scaling",
    "strong_scaling",
    "time_fit",
]

__version__ = "1.0.0"
