# ADR-0006: Diseño del experimento de escalabilidad

**Estado**: Aceptada
**Fecha**: 2026-06-07
**Decisor**: Niels Pacheco

## Contexto

Para sustentar la afirmación de "procesamiento distribuido de Big Data" no basta
entrenar modelos: hay que **medir** cómo escala el sistema. El curso enfatizó la
escalabilidad horizontal frente a la vertical.

## Decisión

Realizar dos benchmarks complementarios:

1. **Escalabilidad de datos** (`data_scaling`): paralelismo fijo, volumen de
   entrenamiento creciente (5%, 10%, 25%, 50%, 100% de ~14.9M filas); se mide el
   tiempo de ajuste. Esperado: crecimiento aproximadamente lineal con el volumen.
2. **Escalabilidad fuerte** (`strong_scaling`): volumen fijo (~3M filas),
   variando los cores del executor local (1, 2, 4, 8); se mide el *speedup*
   respecto a 1 core. Esperado: speedup sublineal por la ley de Amdahl
   (overheads de coordinación y partes seriales).

Cada medición materializa los datos (`count()`/cache) antes de cronometrar para
no incluir la lectura perezosa.

## Opciones consideradas

- Solo medir desempeño predictivo — insuficiente para un artículo de Big Data.
- Variar nodos en un cluster real — descartado por [ADR-0001](0001-pyspark-local-vs-cloud.md);
  la escalabilidad fuerte por cores es el análogo local válido.

## Consecuencias

**Positivas**: evidencia cuantitativa de escalabilidad (curvas de tiempo y
speedup); permite discutir Amdahl y el límite de una sola máquina.

**Negativas**: en local no se observa el costo de red/shuffle entre nodos; el
speedup está acotado por los cores físicos y la fracción serial. Se documenta y
se proyecta el comportamiento en cluster.
