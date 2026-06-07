#!/usr/bin/env python3
"""Experimento principal: comparacion de clasificadores MLlib + escalabilidad de datos.

Flujo (replica el ciclo de vida del dato visto en clase):
    ingesta -> limpieza -> etiqueta -> ingenieria de variables ->
    split -> entrenamiento por modelo -> evaluacion -> benchmark de escalabilidad.

Guarda un JSON con todos los resultados en output/results.json.

Uso:
    python scripts/run_experiments.py --train-rows 2000000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nyc_taxi_mllib import (  # noqa: E402
    add_class_weights,
    add_label,
    build_feature_pipeline,
    build_spark,
    class_balance,
    clean_trips,
    data_scaling,
    evaluate_predictions,
    load_trips,
    make_models,
    time_fit,
)

OUT = Path(__file__).resolve().parents[1] / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-rows", type=int, default=2_000_000,
                    help="Filas (aprox.) para la comparacion de modelos")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cores", default="*")
    ap.add_argument("--driver-memory", default="24g")
    args = ap.parse_args(argv)

    OUT.mkdir(exist_ok=True)
    results: dict = {"config": vars(args).copy()}
    t_start = time.perf_counter()

    def save():
        (OUT / "results.json").write_text(json.dumps(results, indent=2))

    spark = build_spark(app_name="nyc-taxi-experiments", cores=args.cores,
                        driver_memory=args.driver_memory)
    print(">> Spark iniciado:", spark.version)

    # 1. Ingesta + limpieza + etiqueta -----------------------------------
    raw = load_trips(spark)
    n_raw = raw.count()
    prepared = add_label(clean_trips(raw)).cache()
    n_clean = prepared.count()
    bal = class_balance(prepared)
    print(f">> Filas crudas={n_raw:,}  limpias={n_clean:,}  "
          f"pos_rate={bal['pos_rate']:.4f}  ratio={bal['imbalance_ratio']:.1f}:1")
    results["data"] = {"n_raw": n_raw, "n_clean": n_clean, **bal}

    # 2. Ingenieria de variables -----------------------------------------
    weighted = add_class_weights(prepared)
    feat_model = build_feature_pipeline().fit(weighted)
    featurized = feat_model.transform(weighted).select(
        "features", "label", "class_weight"
    ).cache()
    featurized.count()

    # 3. Muestra para comparacion de modelos -----------------------------
    frac = min(1.0, args.train_rows / n_clean)
    sample = featurized.sample(False, frac, seed=args.seed).cache()
    train, test = sample.randomSplit([0.8, 0.2], seed=args.seed)
    train = train.cache(); test = test.cache()
    n_train, n_test = train.count(), test.count()
    print(f">> Comparacion sobre muestra: train={n_train:,}  test={n_test:,}")
    results["sample"] = {"n_train": n_train, "n_test": n_test, "fraction": frac}

    # 4. Entrenamiento + evaluacion por modelo ---------------------------
    models = make_models()
    results["models"] = {}
    for name, est in models.items():
        print(f"   - entrenando {name} ...", flush=True)
        secs, model = time_fit(est, train)
        preds = model.transform(test)
        metrics = evaluate_predictions(preds)
        metrics["train_seconds"] = secs
        results["models"][name] = metrics
        save()  # guardado incremental: no perder la comparacion si falla luego
        print(f"     {name}: AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} "
              f"recall+={metrics['recall_pos']:.4f} t={secs:.1f}s")

    # Liberar cachés de la comparación antes del benchmark pesado.
    train.unpersist(); test.unpersist(); sample.unpersist()

    # 5. Escalabilidad de datos (LogisticRegression, rango completo) ------
    print(">> Benchmark de escalabilidad de datos (LogisticRegression)...")
    lr = make_models(weight_col=None)["LogisticRegression"]
    ds = data_scaling(lr, featurized, fractions=[0.05, 0.1, 0.25, 0.5, 1.0],
                      seed=args.seed)
    results["data_scaling_lr"] = ds
    for r in ds:
        print(f"   n={r['n_rows']:>10,}  t={r['train_seconds']:.1f}s")

    results["wall_clock_seconds"] = time.perf_counter() - t_start
    save()
    print(f">> Resultados guardados en {OUT/'results.json'} "
          f"(total {results['wall_clock_seconds']:.0f}s)")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
