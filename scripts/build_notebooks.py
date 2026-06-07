#!/usr/bin/env python3
"""Construye los notebooks .ipynb del proyecto (EDA y experimentos).

Los notebooks llaman al paquete nyc_taxi_mllib, de modo que la lógica vive en
src/ (probada) y el notebook solo orquesta y narra. Se generan sin ejecutar;
para ejecutarlos: jupyter nbconvert --execute notebooks/01_eda.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(src):
    return nbf.v4.new_code_cell(src)


def build_eda():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# 01 — Análisis Exploratorio de Datos (EDA)\n\n"
           "**Proyecto Big Data — NYC Taxi · Spark MLlib**\n\n"
           "EDA escalable de los viajes de taxi de NYC: estructura, calidad, "
           "desbalance de clases y patrones temporales. Las agregaciones se hacen "
           "en Spark (escalables a todo el volumen) y solo los resúmenes pequeños "
           "se llevan a pandas para graficar."),
        code("import sys; sys.path.insert(0, '../src')\n"
             "from nyc_taxi_mllib import build_spark, load_trips, clean_trips, add_label, class_balance\n"
             "from pyspark.sql import functions as F\n"
             "spark = build_spark('eda-notebook')\n"
             "spark.version"),
        md("## 1. Ingesta\nLectura de los Parquet mensuales (esquema normalizado a tipos canónicos)."),
        code("raw = load_trips(spark)\n"
             "print('Viajes crudos:', raw.count())\n"
             "raw.printSchema()"),
        md("## 2. Limpieza y etiqueta\n"
           "Filtrado de inválidos, solo pagos con tarjeta (veracidad de la propina), "
           "variables temporales/cinemáticas y etiqueta de *propina baja* (<10%)."),
        code("df = add_label(clean_trips(raw)).cache()\n"
             "print('Viajes limpios:', df.count())\n"
             "df.select('trip_distance','trip_duration_min','fare_amount','tip_amount','label').show(5)"),
        md("## 3. Desbalance de clases\n"
           "La clase positiva (propina baja) es minoritaria: el reto central del proyecto."),
        code("bal = class_balance(df)\n"
             "print(bal)\n"
             "print(f\"Propina baja: {bal['pos_rate']:.2%}  |  Desbalance {bal['imbalance_ratio']:.1f}:1\")"),
        md("## 4. Patrón temporal\nTasa de propina baja por hora del día."),
        code("(df.groupBy(F.hour('tpep_pickup_datetime').alias('hora'))\n"
             "   .agg(F.mean('label').alias('tasa_propina_baja'))\n"
             "   .orderBy('hora')).show(24)"),
        md("## 5. Figuras\n"
           "El script `scripts/run_eda.py` genera todas las figuras publication-ready "
           "(paleta Okabe-Ito) en `figures/`. Aquí se invoca el mismo pipeline."),
        code("# Genera figuras fig1..fig6 en ../figures/\n"
             "import subprocess; print(subprocess.run([sys.executable, '../scripts/run_eda.py'],\n"
             "      capture_output=True, text=True).stdout[-400:])"),
        code("spark.stop()"),
    ]
    nbf.write(nb, str(NB / "01_eda.ipynb"))


def build_experiments():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# 02 — Experimentos: comparación de modelos y escalabilidad\n\n"
           "**Proyecto Big Data — NYC Taxi · Spark MLlib**\n\n"
           "Comparación de 4 clasificadores distribuidos (Regresión Logística, Árbol, "
           "Random Forest, GBT) con ponderación de clases, y benchmarks de "
           "escalabilidad. La lógica reside en el paquete `nyc_taxi_mllib`."),
        code("import sys; sys.path.insert(0, '../src')\n"
             "from nyc_taxi_mllib import (build_spark, load_trips, clean_trips, add_label,\n"
             "    add_class_weights, build_feature_pipeline, make_models, time_fit,\n"
             "    evaluate_predictions, class_balance, data_scaling)\n"
             "spark = build_spark('experiments-notebook')"),
        md("## 1. Preparación de datos y variables\n"
           "Limpieza → etiqueta → ponderación de clases → pipeline de features "
           "(`StringIndexer` + `OneHotEncoder` + `VectorAssembler`)."),
        code("prepared = add_class_weights(add_label(clean_trips(load_trips(spark)))).cache()\n"
             "print(class_balance(prepared))\n"
             "feat_model = build_feature_pipeline().fit(prepared)\n"
             "featurized = feat_model.transform(prepared).select('features','label','class_weight').cache()\n"
             "featurized.count()"),
        md("## 2. Muestra y partición\nMuestra de ~2M viajes, split 80/20 idéntico para todos los modelos."),
        code("sample = featurized.sample(False, 2_000_000/prepared.count(), seed=42).cache()\n"
             "train, test = sample.randomSplit([0.8, 0.2], seed=42)\n"
             "train.cache().count(), test.cache().count()"),
        md("## 3. Entrenamiento y evaluación de los 4 modelos\n"
           "Métricas: AUC, F1, accuracy y **recall de la clase minoritaria** (la métrica honesta)."),
        code("results = {}\n"
             "for name, est in make_models().items():\n"
             "    secs, model = time_fit(est, train)\n"
             "    m = evaluate_predictions(model.transform(test)); m['train_seconds'] = secs\n"
             "    results[name] = m\n"
             "    print(f\"{name:18s} AUC={m['auc']:.3f} F1={m['f1']:.3f} \"\n"
             "          f\"acc={m['accuracy']:.3f} recall+={m['recall_pos']:.3f} t={secs:.1f}s\")"),
        md("> **Hallazgo:** la *accuracy* engaña bajo desbalance (un trivial daría ~90.6%). "
           "GBT y Random Forest dan el mejor AUC; la elección depende del costo de cómputo."),
        md("## 4. Escalabilidad de datos\nTiempo de entrenamiento de la Regresión Logística vs. volumen."),
        code("lr = make_models(weight_col=None)['LogisticRegression']\n"
             "for r in data_scaling(lr, featurized, fractions=[0.05, 0.1, 0.25, 0.5, 1.0], seed=42):\n"
             "    print(f\"n={r['n_rows']:>10,}  t={r['train_seconds']:.1f}s\")"),
        md("La escalabilidad fuerte (vs. núcleos) está en `scripts/run_strong_scaling.py` "
           "(requiere recrear la sesión Spark con distinto nº de cores)."),
        code("spark.stop()"),
    ]
    nbf.write(nb, str(NB / "02_experiments.ipynb"))


def main():
    NB.mkdir(exist_ok=True)
    build_eda()
    build_experiments()
    print("Notebooks generados: notebooks/01_eda.ipynb, notebooks/02_experiments.ipynb")


if __name__ == "__main__":
    main()
