#!/usr/bin/env python3
"""Diagrama de la arquitectura / ciclo de vida del dato (vectorial, reproducible).

Genera figures/fig7_arquitectura.(png|pdf): un pipeline por etapas que refleja
el ciclo de vida del dato visto en clase, mapeando cada etapa a su tecnologia y
a las V de Big Data que aborda.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path(__file__).resolve().parents[1] / "figures"
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7"]

STAGES = [
    ("Ingesta", "NYC TLC\nParquet (6 meses)", "Volumen, Variedad", OKABE_ITO[0]),
    ("Almacenamiento", "Data Lake local\n(Parquet columnar)", "Volumen", OKABE_ITO[4]),
    ("Procesamiento\ndistribuido", "Apache Spark\n(local[*])", "Velocidad", OKABE_ITO[2]),
    ("Limpieza +\nFeatures", "MLlib Pipeline\nStringIndexer,\nVectorAssembler", "Veracidad", OKABE_ITO[3]),
    ("Modelado", "LR / DT / RF /\nGBT (MLlib)", "Valor", OKABE_ITO[1]),
    ("Evaluacion +\nViz", "AUC, F1, recall\nfiguras Okabe-Ito", "Visualizacion", OKABE_ITO[5]),
]


def main() -> int:
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, len(STAGES) * 2); ax.set_ylim(0, 3); ax.axis("off")

    box_w, box_h, y0 = 1.7, 1.4, 1.0
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})

    for i, (title, tech, vs, color) in enumerate(STAGES):
        x = i * 2 + 0.15
        box = FancyBboxPatch((x, y0), box_w, box_h,
                             boxstyle="round,pad=0.03,rounding_size=0.12",
                             linewidth=1.5, edgecolor=color, facecolor=color + "22")
        ax.add_patch(box)
        cx = x + box_w / 2
        ax.text(cx, y0 + box_h - 0.28, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=color)
        ax.text(cx, y0 + box_h / 2 - 0.12, tech, ha="center", va="center", fontsize=7.6)
        # Etiqueta de V de Big Data bajo cada etapa
        ax.text(cx, y0 - 0.28, vs, ha="center", va="center", fontsize=7,
                style="italic", color="#444444")
        if i < len(STAGES) - 1:
            arrow = FancyArrowPatch((x + box_w, y0 + box_h / 2),
                                    (x + 2 + 0.15, y0 + box_h / 2),
                                    arrowstyle="-|>", mutation_scale=14,
                                    linewidth=1.4, color="#666666")
            ax.add_patch(arrow)

    ax.text(len(STAGES), 2.85, "Ciclo de vida del dato — Prediccion de propina baja (NYC Taxi, Spark MLlib)",
            ha="center", va="center", fontsize=11, fontweight="bold")
    handle = mpatches.Patch(color="white", label="Etiquetas en cursiva: V de Big Data abordada en cada etapa")
    ax.legend(handles=[handle], loc="lower center", frameon=False, fontsize=7.5,
              bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(FIG / "fig7_arquitectura.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / "fig7_arquitectura.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Arquitectura -> figures/fig7_arquitectura.png|pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
