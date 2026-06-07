# ADR-0001: PySpark en modo local en lugar de cloud (EMR/Databricks)

**Estado**: Aceptada
**Fecha**: 2026-06-07
**Decisor**: Niels Pacheco

## Contexto

El proyecto requiere demostrar procesamiento distribuido de Big Data sobre
decenas de millones de registros. El curso mostró clusters reales en AWS EMR
(Clase 4). Sin embargo, el entregable debe ser **reproducible por el docente y
los compañeros sin incurrir en costos ni credenciales**, y dentro del plazo del
curso.

## Decisión

Ejecutar Apache Spark en **modo local** (`local[*]`), que crea un executor con
múltiples hilos sobre la máquina del autor, simulando el paralelismo de un
cluster. Se documenta cómo escalaría la misma arquitectura en EMR/Databricks.

## Opciones consideradas

- **PySpark local** — Gratis, reproducible (`pip install pyspark`), suficiente
  para ejercitar particiones, `Pipeline`, MLlib y medir escalabilidad fuerte
  variando cores. Limitado a una máquina.
- **AWS EMR / Databricks real** — Más fiel a producción y mayor volumen, pero
  implica costos, gestión de credenciales y no es reproducible sin una cuenta.

## Consecuencias

**Positivas**: reproducibilidad total, costo cero, mismo API de Spark que en
cluster (el código migra sin cambios a EMR). Permite el experimento de
escalabilidad fuerte (1→8 cores).

**Negativas**: el volumen está acotado por la RAM/CPU de una máquina; no se
evalúa la latencia de red ni el shuffle entre nodos. Se mitiga documentando
explícitamente esta limitación y proyectando el comportamiento en cluster.

## Relacionadas

- [ADR-0006](0006-experimento-escalabilidad.md): el diseño de escalabilidad
  depende de esta decisión.
