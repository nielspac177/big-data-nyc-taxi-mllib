#!/usr/bin/env python3
"""Figuras de resultados a partir de output/results.json y output/scaling.json.

No requiere Spark: lee los resultados ya calculados y produce figuras
publication-ready (paleta Okabe-Ito).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "output"

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 220, "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
})


def fig_model_comparison(res: dict) -> None:
    models = list(res["models"].keys())
    metrics = ["auc", "f1", "recall_pos", "accuracy"]
    labels = ["AUC", "F1", "Recall\n(propina baja)", "Accuracy"]
    x = np.arange(len(models)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        vals = [res["models"][md][m] for md in models]
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=OKABE_ITO[i])
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("Valor de la métrica"); ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.set_title("Comparación de modelos (test, con ponderación de clases)")
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(FIG / "fig8_comparacion_modelos.png")
    fig.savefig(FIG / "fig8_comparacion_modelos.pdf"); plt.close(fig)


def fig_perf_vs_cost(res: dict) -> None:
    models = list(res["models"].keys())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, md in enumerate(models):
        m = res["models"][md]
        ax.scatter(m["train_seconds"], m["auc"], s=90, color=OKABE_ITO[i], zorder=3)
        ax.annotate(md, (m["train_seconds"], m["auc"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Tiempo de entrenamiento (s)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Trade-off desempeño vs. costo computacional")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig9_tradeoff.png")
    fig.savefig(FIG / "fig9_tradeoff.pdf"); plt.close(fig)


def fig_data_scaling(res: dict) -> None:
    ds = res["data_scaling_lr"]
    n = [d["n_rows"] / 1e6 for d in ds]
    t = [d["train_seconds"] for d in ds]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(n, t, marker="o", color=OKABE_ITO[0], label="Tiempo medido")
    # referencia lineal anclada al primer punto
    ratio = t[0] / n[0]
    ax.plot(n, [ratio * x for x in n], ls="--", color="grey", label="Lineal ideal")
    ax.set_xlabel("Datos de entrenamiento (millones de filas)")
    ax.set_ylabel("Tiempo de entrenamiento (s)")
    ax.set_title("Escalabilidad de datos (LogisticRegression)")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "fig10_escalabilidad_datos.png")
    fig.savefig(FIG / "fig10_escalabilidad_datos.pdf"); plt.close(fig)


def fig_strong_scaling(scaling: dict) -> None:
    rs = scaling["results"]
    cores = [r["cores"] for r in rs]
    speed = [r["speedup"] for r in rs]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(cores, speed, marker="o", color=OKABE_ITO[1], label="Speedup medido")
    ax.plot(cores, cores, ls="--", color="grey", label="Speedup ideal (lineal)")
    ax.set_xlabel("Número de cores"); ax.set_ylabel("Speedup (vs. 1 core)")
    ax.set_xticks(cores)
    ax.set_title(f"Escalabilidad fuerte (LogisticRegression, {scaling['rows']/1e6:.0f}M filas)")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "fig11_escalabilidad_fuerte.png")
    fig.savefig(FIG / "fig11_escalabilidad_fuerte.pdf"); plt.close(fig)


def main() -> int:
    FIG.mkdir(exist_ok=True)
    res = json.load(open(OUT / "results.json"))
    fig_model_comparison(res)
    fig_perf_vs_cost(res)
    fig_data_scaling(res)
    scaling_path = OUT / "scaling.json"
    if scaling_path.exists():
        fig_strong_scaling(json.load(open(scaling_path)))
        print("Figuras de resultados (incl. escalabilidad fuerte) -> figures/")
    else:
        print("Figuras de resultados -> figures/ (falta scaling.json para fig11)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
