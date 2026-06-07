# ADR-0002: Usar NYC TLC Yellow Taxi como dataset

**Estado**: Aceptada
**Fecha**: 2026-06-07
**Decisor**: Niels Pacheco

## Contexto

La modalidad de artículo científico requiere un dataset público y de gran
volumen para comparar algoritmos de ML sobre Big Data. Debe ser fácilmente
descargable, con suficiente volumen y con una narrativa de negocio clara.

## Decisión

Usar los **NYC TLC Trip Record Data (Yellow Taxi)**: archivos Parquet mensuales
públicos, ~3M de viajes por mes. Se emplean 6 meses de 2023 (~19.5M de viajes
crudos, ~14.9M tras limpieza).

## Opciones consideradas

- **NYC Taxi** — Iconico, formato Parquet columnar (alineado a Big Data),
  decenas de millones de filas, fuerte narrativa de negocio (pricing, propinas,
  demanda). Esquema con ligera deriva entre meses (manejable).
- **US Flight Delays** — Buen volumen pero menos variables de negocio para la
  tarea de propina.
- **HIGGS / benchmark sintético** — Limpio pero sin narrativa de negocio.

## Consecuencias

**Positivas**: volumen real, formato columnar eficiente, dominio comprensible,
descarga reproducible (`scripts/download_data.py`).

**Negativas**: deriva de esquema entre meses (INT vs BIGINT/DOUBLE) que obliga a
castear columnas a tipos canónicos al ingerir (ver `ingest.py`). Las propinas en
efectivo no se registran → ver [ADR-0003](0003-objetivo-propina-baja.md).
