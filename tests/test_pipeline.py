"""Smoke tests del pipeline sobre una muestra pequena de datos reales.

Se omiten automaticamente si los Parquet de NYC TLC aun no se han descargado
(ejecutar antes ``python scripts/download_data.py``).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nyc_taxi_mllib import (
    add_class_weights,
    add_label,
    build_feature_pipeline,
    build_spark,
    class_balance,
    clean_trips,
    evaluate_predictions,
    make_models,
    time_fit,
)
from nyc_taxi_mllib.ingest import RAW_DIR

pytestmark = pytest.mark.skipif(
    not list(RAW_DIR.glob("*.parquet")),
    reason="No hay datos en data/raw/ (ejecutar scripts/download_data.py)",
)


@pytest.fixture(scope="module")
def spark():
    s = build_spark("pytest", cores=2)
    yield s
    s.stop()


@pytest.fixture(scope="module")
def sample(spark):
    f = sorted(RAW_DIR.glob("*.parquet"))[0]
    raw = spark.read.parquet(str(f))
    return add_label(clean_trips(raw)).sample(False, 0.02, seed=1).cache()


def test_clean_produces_required_columns(sample):
    cols = set(sample.columns)
    assert {"label", "trip_duration_min", "avg_speed_mph", "pickup_hour"} <= cols


def test_label_is_binary(sample):
    labels = {r["label"] for r in sample.select("label").distinct().collect()}
    assert labels <= {0, 1}


def test_class_balance_minority_is_low_tip(sample):
    bal = class_balance(sample)
    assert bal["total"] > 0
    assert 0.0 < bal["pos_rate"] < 0.5  # propina baja es minoritaria


def test_end_to_end_training(sample):
    weighted = add_class_weights(sample)
    feat = build_feature_pipeline().fit(weighted).transform(weighted)
    feat = feat.select("features", "label", "class_weight")
    train, test = feat.randomSplit([0.8, 0.2], seed=1)
    model_def = make_models()["LogisticRegression"]
    secs, model = time_fit(model_def, train)
    metrics = evaluate_predictions(model.transform(test))
    assert secs > 0
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["recall_pos"] <= 1.0
