# ADR-0005: Ponderación de clases para el desbalance ~10:1

**Estado**: Aceptada
**Fecha**: 2026-06-07
**Decisor**: Niels Pacheco

## Contexto

La clase positiva (propina baja) representa ~9.4% de los datos (desbalance
~9.7:1). Sin tratamiento, un clasificador trivial que prediga siempre "propina
normal" alcanza ~90% de *accuracy* pero **0% de recall** sobre la clase de
interés — exactamente la trampa señalada en clase con el caso de cierres de
restaurantes.

## Decisión

Aplicar **ponderación de instancias por clase** (`weightCol`): a cada viaje se
le asigna un peso `total / (2 · n_clase)`, de modo que ambas clases contribuyen
por igual a la función de pérdida. Se reportan métricas honestas bajo desbalance:
**AUC**, **F1** y **recall de la clase minoritaria**, además de la *accuracy*.

## Opciones consideradas

- **Ponderación de clases (`weightCol`)** — Nativa en LR/DT/RF, sin alterar el
  tamaño de datos, eficiente en distribuido. Elegida.
- **Oversampling / undersampling** — Vista en clase; aumenta o reduce el volumen
  y el costo. Se menciona como alternativa y se deja como trabajo futuro
  comparar su efecto.
- **Ajuste de umbral** — Complementaria; se discute pero no se barre
  exhaustivamente.

## Consecuencias

**Positivas**: mejora el recall de la minoría sin duplicar datos; métricas de
evaluación honestas; alineado con el énfasis del curso.

**Negativas**: `GBTClassifier` no soporta `weightCol` en Spark 3.5 → queda sin
ponderar y con recall bajo de la minoría, lo que se reporta explícitamente como
limitación del ecosistema (no del enfoque).
