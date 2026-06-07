"""Ingesta de los registros de viajes (Yellow Taxi) de NYC TLC en Spark.

Los archivos Parquet mensuales de NYC TLC presentan *deriva de esquema*: una
misma columna puede ser INT en un mes y BIGINT/DOUBLE en otro. Para obtener un
DataFrame unico y consistente se lee cada archivo por separado, se castean las
columnas necesarias a tipos canonicos y se unen con ``unionByName``.
"""
from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

__all__ = ["RAW_DIR", "CANONICAL_COLUMNS", "load_trips"]

# Directorio por defecto donde scripts/download_data.py deja los Parquet.
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Columna -> tipo canonico al que se castea tras la lectura.
CANONICAL_COLUMNS: dict[str, str] = {
    "tpep_pickup_datetime": "timestamp",
    "tpep_dropoff_datetime": "timestamp",
    "passenger_count": "double",
    "trip_distance": "double",
    "RatecodeID": "double",
    "PULocationID": "int",
    "DOLocationID": "int",
    "payment_type": "int",
    "fare_amount": "double",
    "extra": "double",
    "tip_amount": "double",
    "tolls_amount": "double",
    "total_amount": "double",
    "congestion_surcharge": "double",
    "airport_fee": "double",
}


def _read_one(spark: SparkSession, path: Path) -> DataFrame:
    df = spark.read.parquet(str(path))
    return df.select(
        *[F.col(c).cast(t).alias(c) for c, t in CANONICAL_COLUMNS.items()]
    )


def load_trips(spark: SparkSession, path: str | Path = RAW_DIR) -> DataFrame:
    """Lee todos los Parquet mensuales de ``path`` como un unico DataFrame.

    Cada fila es un viaje; las columnas se normalizan a ``CANONICAL_COLUMNS``.
    """
    path = Path(path)
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No se encontraron Parquet en {path}")
    df = _read_one(spark, files[0])
    for f in files[1:]:
        df = df.unionByName(_read_one(spark, f))
    return df
