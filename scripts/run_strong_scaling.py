#!/usr/bin/env python3
"""Experimento de escalabilidad fuerte: tiempo de entrenamiento vs nº de cores.

Se fija el volumen de datos y se entrena el mismo modelo (LogisticRegression)
variando el numero de cores del executor local (1, 2, 4, 8). El *speedup* se
mide respecto a 1 core. Es la evidencia central de procesamiento distribuido.

Para que cada configuracion sea comparable, primero se materializa una muestra
featurizada en data/processed/ y luego cada sesion Spark (con distinto nº de
cores) la relee y entrena.

Uso:
    python scripts/run_strong_scaling.py --rows 3000000 --cores 1 2 4 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nyc_taxi_mllib import (  # noqa: E402
    add_class_weights,
    add_label,
    build_feature_pipeline,
    build_spark,
    clean_trips,
    load_trips,
    make_models,
    strong_scaling,
)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT = Path(__file__).resolve().parents[1] / "output"


def prepare_sample(rows: int, seed: int) -> str:
    """Materializa una muestra featurizada en Parquet y devuelve su ruta."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / f"featurized_{rows}.parquet"
    if out_path.exists():
        print(f">> Muestra ya existe: {out_path}")
        return str(out_path)
    spark = build_spark("prep-sample", cores="*")
    prepared = add_class_weights(add_label(clean_trips(load_trips(spark))))
    n = prepared.count()
    frac = min(1.0, rows / n)
    sample = prepared.sample(False, frac, seed=seed)
    feat = build_feature_pipeline().fit(sample).transform(sample)
    feat.select("features", "label", "class_weight").write.mode("overwrite").parquet(str(out_path))
    print(f">> Muestra featurizada escrita ({frac*n:,.0f} filas aprox) -> {out_path}")
    spark.stop()
    return str(out_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3_000_000)
    ap.add_argument("--cores", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)

    sample_path = prepare_sample(args.rows, args.seed)

    def build_train(spark):
        return spark.read.parquet(sample_path)

    def build_estimator(spark):
        return make_models(weight_col=None)["LogisticRegression"]

    results = strong_scaling(build_estimator, build_train, args.cores)
    payload = {"rows": args.rows, "results": results}
    (OUT / "scaling.json").write_text(json.dumps(payload, indent=2))
    print(">> Escalabilidad fuerte:")
    for r in results:
        print(f"   cores={r['cores']}  t={r['train_seconds']:.1f}s  speedup={r['speedup']:.2f}x")
    print(f">> Guardado en {OUT/'scaling.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
