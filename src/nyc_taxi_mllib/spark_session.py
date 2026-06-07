"""Fabrica de :class:`SparkSession` para ejecucion local reproducible.

Encapsula la configuracion del cluster *local* que simula el procesamiento
distribuido: numero de cores, memoria del driver y el ajuste necesario para
leer los Parquet de NYC TLC (cuyo esquema fisico de ``passenger_count`` varia
entre meses, lo que obliga a desactivar el lector vectorizado).
"""
from __future__ import annotations

import os

from pyspark.sql import SparkSession

__all__ = ["build_spark"]


def build_spark(
    app_name: str = "nyc-taxi-mllib",
    cores: int | str = "*",
    driver_memory: str = "8g",
    shuffle_partitions: int | None = None,
) -> SparkSession:
    """Construye (o recupera) una ``SparkSession`` en modo local.

    Parameters
    ----------
    app_name:
        Nombre de la aplicacion Spark.
    cores:
        Numero de cores del executor local. ``"*"`` usa todos los disponibles;
        un entero fija el paralelismo (util para los experimentos de
        escalabilidad fuerte).
    driver_memory:
        Memoria asignada al driver (todo corre en el driver en modo local).
    shuffle_partitions:
        Numero de particiones para operaciones de shuffle. Si es ``None`` se
        deja un valor moderado adecuado a una sola maquina.
    """
    # Spark necesita una IP de loopback estable en macOS.
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    builder = (
        SparkSession.builder.master(f"local[{cores}]")
        .appName(app_name)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.memory", driver_memory)
        # Los archivos mensuales de NYC TLC mezclan INT64/DOUBLE en columnas
        # como passenger_count; el lector no vectorizado tolera la conversion.
        # (No se usa mergeSchema porque algunos meses difieren INT/BIGINT y la
        #  union estricta de esquemas fallaria; el esquema de un archivo basta.)
        .config("spark.sql.parquet.enableVectorizedReader", "false")
    )
    if shuffle_partitions is not None:
        builder = builder.config("spark.sql.shuffle.partitions", str(shuffle_partitions))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
