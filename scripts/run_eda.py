#!/usr/bin/env python3
"""Analisis exploratorio (EDA) escalable + figuras publication-ready.

Las agregaciones se hacen en Spark (escalable a todo el volumen) y solo los
resumenes pequenos se traen a pandas para graficar. Estilo: paleta Okabe-Ito
(colorblind-safe), coherente con el examen parcial.

Genera figuras en figures/ y un resumen numerico en output/eda.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pyspark.sql import functions as F  # noqa: E402

from nyc_taxi_mllib import (  # noqa: E402
    LOW_TIP_THRESHOLD,
    add_label,
    build_spark,
    class_balance,
    clean_trips,
    load_trips,
)
from nyc_taxi_mllib.features import NUMERIC_FEATURES  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
OUT = Path(__file__).resolve().parents[1] / "output"

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00",
             "#56B4E9", "#CC79A7", "#F0E442", "#000000"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 220,
    "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
})
sns.set_palette(OKABE_ITO)


def main() -> int:
    FIG.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    res: dict = {}
    spark = build_spark("nyc-taxi-eda", cores="*")

    raw = load_trips(spark)
    res["n_raw"] = raw.count()
    df = add_label(clean_trips(raw)).withColumn(
        "tip_pct", F.col("tip_amount") / F.col("fare_amount")
    ).cache()
    res["n_clean"] = df.count()
    res["threshold"] = LOW_TIP_THRESHOLD
    res["balance"] = class_balance(df)

    # --- Fig 1: volumen por mes (historia de Volumen) -------------------
    by_month = (df.groupBy(F.date_format("tpep_pickup_datetime", "yyyy-MM").alias("month"))
                .count().orderBy("month").toPandas())
    by_month = by_month[by_month["month"].str.startswith("2023")]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(by_month["month"], by_month["count"] / 1e6, color=OKABE_ITO[0])
    ax.set_ylabel("Viajes (millones)"); ax.set_xlabel("Mes (2023)")
    ax.set_title("Volumen mensual de viajes (Yellow Taxi, pago con tarjeta)")
    plt.xticks(rotation=45, ha="right"); fig.tight_layout()
    fig.savefig(FIG / "fig1_volumen_mensual.png"); plt.close(fig)
    res["rows_per_month"] = by_month.to_dict("records")

    # --- Fig 2: distribucion del % de propina + umbral ------------------
    hist = (df.where(F.col("tip_pct") <= 0.6)
            .select((F.floor(F.col("tip_pct") / 0.02) * 0.02).alias("bin"))
            .groupBy("bin").count().orderBy("bin").toPandas())
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(hist["bin"], hist["count"] / 1e6, width=0.018,
           color=OKABE_ITO[4], edgecolor="white", linewidth=0.3)
    ax.axvline(LOW_TIP_THRESHOLD, color=OKABE_ITO[1], ls="--", lw=1.5,
               label=f"Umbral propina baja ({LOW_TIP_THRESHOLD:.0%})")
    ax.set_xlabel("Propina / Tarifa"); ax.set_ylabel("Viajes (millones)")
    ax.set_title("Distribucion del porcentaje de propina")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "fig2_distribucion_propina.png"); plt.close(fig)

    # --- Fig 3: balance de clases ---------------------------------------
    bal = res["balance"]
    fig, ax = plt.subplots(figsize=(3.6, 3))
    bars = ax.bar(["Propina normal\n(>=10%)", "Propina baja\n(<10%)"],
                  [bal["n_neg"] / 1e6, bal["n_pos"] / 1e6],
                  color=[OKABE_ITO[0], OKABE_ITO[1]])
    for b, v in zip(bars, [bal["n_neg"], bal["n_pos"]]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v/1e6:.2f}M\n({v/bal['total']:.1%})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Viajes (millones)")
    ax.set_title(f"Desbalance de clases ({bal['imbalance_ratio']:.1f}:1)")
    ax.set_ylim(0, bal["n_neg"] / 1e6 * 1.18); fig.tight_layout()
    fig.savefig(FIG / "fig3_balance_clases.png"); plt.close(fig)

    # --- Fig 4: tasa de propina baja por hora y dia ---------------------
    by_hour = (df.groupBy(F.hour("tpep_pickup_datetime").alias("h"))
               .agg(F.mean("label").alias("low_rate")).orderBy("h").toPandas())
    by_dow = (df.groupBy(F.dayofweek("tpep_pickup_datetime").alias("d"))
              .agg(F.mean("label").alias("low_rate")).orderBy("d").toPandas())
    dow_names = ["Dom", "Lun", "Mar", "Mie", "Jue", "Vie", "Sab"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3))
    axes[0].plot(by_hour["h"], by_hour["low_rate"] * 100, marker="o", ms=3, color=OKABE_ITO[1])
    axes[0].set_xlabel("Hora del dia"); axes[0].set_ylabel("Tasa propina baja (%)")
    axes[0].set_title("Por hora")
    axes[1].bar([dow_names[int(d) - 1] for d in by_dow["d"]], by_dow["low_rate"] * 100, color=OKABE_ITO[2])
    axes[1].set_ylabel("Tasa propina baja (%)"); axes[1].set_title("Por dia de la semana")
    fig.tight_layout(); fig.savefig(FIG / "fig4_propina_temporal.png"); plt.close(fig)
    res["low_rate_by_hour"] = by_hour.to_dict("records")

    # --- Fig 5: correlacion de variables numericas ----------------------
    num_cols = [c for c in NUMERIC_FEATURES if c not in ("PULocationID", "DOLocationID")]
    sample_pd = df.select(*num_cols, "label").sample(False, 0.02, seed=7).toPandas()
    corr = sample_pd.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 6}, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Matriz de correlacion (variables numericas + etiqueta)")
    fig.tight_layout(); fig.savefig(FIG / "fig5_correlacion.png"); plt.close(fig)

    # --- Fig 6: distribuciones por clase (distancia, duracion, tarifa) --
    feats = ["trip_distance", "trip_duration_min", "fare_amount"]
    sp = df.select(*feats, "label").where(
        (F.col("trip_distance") < 25) & (F.col("trip_duration_min") < 80) & (F.col("fare_amount") < 120)
    ).sample(False, 0.01, seed=11).toPandas()
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))
    titles = ["Distancia (millas)", "Duracion (min)", "Tarifa (USD)"]
    for ax, f, t in zip(axes, feats, titles):
        for lab, col, name in [(0, OKABE_ITO[0], "normal"), (1, OKABE_ITO[1], "baja")]:
            ax.hist(sp[sp.label == lab][f], bins=40, density=True, alpha=0.55,
                    color=col, label=f"propina {name}")
        ax.set_xlabel(t); ax.set_yticks([])
    axes[0].legend(fontsize=7); fig.suptitle("Distribuciones por clase", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "fig6_distribuciones_clase.png"); plt.close(fig)

    (OUT / "eda.json").write_text(json.dumps(res, indent=2, default=str))
    print("EDA OK -> figuras en figures/, resumen en output/eda.json")
    print("balance:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in bal.items()})
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
