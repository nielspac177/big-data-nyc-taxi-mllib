# Architecture Decision Records (ADR)

Registro de las decisiones de diseño significativas del proyecto *Predicción de
propina baja en viajes de taxi de NYC con Spark MLlib*.

## Índice

| ADR | Título | Estado | Fecha |
| --- | --- | --- | --- |
| [0001](0001-pyspark-local-vs-cloud.md) | PySpark local en lugar de cloud (EMR/Databricks) | Aceptada | 2026-06-07 |
| [0002](0002-dataset-nyc-taxi.md) | Usar NYC TLC Yellow Taxi como dataset | Aceptada | 2026-06-07 |
| [0003](0003-objetivo-propina-baja.md) | Objetivo: predecir propina baja (clase minoritaria) | Aceptada | 2026-06-07 |
| [0004](0004-algoritmos-comparados.md) | Comparar LR, Decision Tree, Random Forest y GBT | Aceptada | 2026-06-07 |
| [0005](0005-manejo-desbalance.md) | Ponderación de clases para el desbalance 10:1 | Aceptada | 2026-06-07 |
| [0006](0006-experimento-escalabilidad.md) | Diseño del experimento de escalabilidad | Aceptada | 2026-06-07 |

## Estados

- **Propuesta**: en discusión.
- **Aceptada**: decisión tomada e implementada.
- **Deprecada / Superada**: reemplazada por otra ADR.
