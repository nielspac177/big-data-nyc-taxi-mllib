# ADR-0004: Comparar LR, Decision Tree, Random Forest y GBT

**Estado**: Aceptada
**Fecha**: 2026-06-07
**Decisor**: Niels Pacheco

## Contexto

La pregunta de investigación es qué algoritmo distribuido ofrece el mejor
*trade-off* entre desempeño y costo computacional. Se necesitan algoritmos que
(a) estén en Spark MLlib, (b) hayan sido cubiertos en clase y (c) cubran un
espectro de complejidad (lineal → ensambles).

## Decisión

Comparar cuatro clasificadores de `pyspark.ml.classification`:

1. **LogisticRegression** — línea base lineal, rápida, interpretable.
2. **DecisionTreeClassifier** — no lineal, interpretable.
3. **RandomForestClassifier** — ensamble bagging, robusto.
4. **GBTClassifier** — ensamble boosting, suele maximizar desempeño.

Todos comparten el mismo vector de `features` y partición train/test, de modo
que la comparación es justa.

## Opciones consideradas

- Incluir Naive Bayes / LinearSVC — descartados para acotar el alcance; LR ya
  cubre el caso lineal y NB asume independencia poco realista aquí.
- Solo LR vs un ensamble — insuficiente para una comparación rica.

## Consecuencias

**Positivas**: barre el espectro lineal↔ensamble; permite discutir el
*trade-off* desempeño/tiempo (los ensambles cuestan más).

**Negativas**: `GBTClassifier` no soporta `weightCol` en Spark 3.5, por lo que
no recibe ponderación de clases y su recall de la minoría es bajo; se documenta
como hallazgo y limitación. Ver [ADR-0005](0005-manejo-desbalance.md).
