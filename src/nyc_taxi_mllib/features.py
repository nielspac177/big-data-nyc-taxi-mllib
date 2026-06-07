"""Limpieza, ingenieria de variables y etiqueta para la tarea *low-tip*.

La tarea es de clasificacion binaria con **fuerte desbalance**: predecir si un
viaje pagado con tarjeta de credito dejara una **propina baja**
(``tip_amount / fare_amount < 0.10``). La clase positiva (propina baja) es
minoritaria (~9-10% de los viajes con tarjeta), lo que reproduce el escenario
desbalanceado enfatizado en clase, donde la *accuracy* es enganosa y hay que
mirar AUC y recall de la clase minoritaria.

Decision critica de calidad de datos (veracidad): las propinas en efectivo NO
se registran en el dataset, por lo que el analisis se restringe a pagos con
tarjeta (``payment_type == 1``). Ademas, para evitar *fuga de objetivo*, se
EXCLUYEN como predictores ``tip_amount`` y ``total_amount`` (este ultimo
contiene la propina por construccion).
"""
from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

__all__ = [
    "LOW_TIP_THRESHOLD",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "clean_trips",
    "add_label",
    "build_feature_pipeline",
]

LOW_TIP_THRESHOLD = 0.10

# Predictores numericos (ninguno derivado de la propina -> sin fuga de objetivo).
NUMERIC_FEATURES = [
    "trip_distance",
    "passenger_count",
    "trip_duration_min",
    "avg_speed_mph",
    "fare_amount",
    "extra",
    "tolls_amount",
    "congestion_surcharge",
    "airport_fee",
    "pickup_hour",
    "is_weekend",
    "is_night",
    "PULocationID",
    "DOLocationID",
]

# Predictores categoricos (cardinalidad baja -> one-hot).
CATEGORICAL_FEATURES = ["pickup_dow", "RatecodeID"]


def clean_trips(df: DataFrame) -> DataFrame:
    """Filtra registros invalidos y deriva variables temporales/cinematicas.

    Reglas de limpieza (acordes al ciclo de vida del dato visto en clase:
    deteccion de outliers/ruido + transformacion):
      * Solo pagos con tarjeta (``payment_type == 1``) -> la propina es veraz.
      * ``fare_amount`` y ``trip_distance`` estrictamente positivos y acotados.
      * ``passenger_count`` entre 1 y 6.
      * Duracion del viaje entre 1 y 180 minutos.
      * Velocidad media plausible (0 < v <= 100 mph).
    """
    pickup, dropoff = "tpep_pickup_datetime", "tpep_dropoff_datetime"

    out = (
        df.where(F.col("payment_type") == 1)
        .withColumn(
            "trip_duration_min",
            (
                F.col(dropoff).cast("timestamp").cast("long")
                - F.col(pickup).cast("timestamp").cast("long")
            )
            / 60.0,
        )
        .where((F.col("fare_amount") > 0) & (F.col("fare_amount") < 500))
        .where((F.col("trip_distance") > 0) & (F.col("trip_distance") < 100))
        .where((F.col("passenger_count") >= 1) & (F.col("passenger_count") <= 6))
        .where((F.col("trip_duration_min") >= 1) & (F.col("trip_duration_min") <= 180))
        .withColumn("avg_speed_mph", F.col("trip_distance") / (F.col("trip_duration_min") / 60.0))
        .where((F.col("avg_speed_mph") > 0) & (F.col("avg_speed_mph") <= 100))
        # Variables temporales
        .withColumn("pickup_hour", F.hour(pickup))
        .withColumn("pickup_dow", F.dayofweek(pickup))  # 1=Dom ... 7=Sab
        .withColumn("is_weekend", (F.dayofweek(pickup).isin(1, 7)).cast("int"))
        .withColumn("is_night", ((F.hour(pickup) >= 22) | (F.hour(pickup) <= 5)).cast("int"))
        # Imputacion de surcharges nulos a 0 (no aplicables a todos los viajes)
        .fillna(0, subset=["extra", "tolls_amount", "congestion_surcharge", "airport_fee"])
        .fillna(1, subset=["RatecodeID"])
    )
    return out


def add_label(df: DataFrame, threshold: float = LOW_TIP_THRESHOLD) -> DataFrame:
    """Agrega la columna binaria ``label`` = propina baja (< ``threshold``).

    La clase positiva (1) es el viaje con propina baja (minoritaria).
    """
    ratio = F.col("tip_amount") / F.col("fare_amount")
    return df.withColumn("label", (ratio < F.lit(threshold)).cast("int"))


def build_feature_pipeline() -> Pipeline:
    """Pipeline MLlib (sin entrenar) que produce la columna ``features``.

    Replica el patron de clase: ``StringIndexer`` -> ``OneHotEncoder`` para las
    categoricas y ``VectorAssembler`` para ensamblar el vector final.
    """
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in CATEGORICAL_FEATURES
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
        for c in CATEGORICAL_FEATURES
    ]
    assembler = VectorAssembler(
        inputCols=NUMERIC_FEATURES + [f"{c}_ohe" for c in CATEGORICAL_FEATURES],
        outputCol="features",
        handleInvalid="skip",
    )
    return Pipeline(stages=[*indexers, *encoders, assembler])
