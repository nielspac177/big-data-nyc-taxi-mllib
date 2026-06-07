#!/usr/bin/env python3
"""Descarga reproducible de los registros de viajes (Yellow Taxi) de NYC TLC.

Los datos son publicos y se distribuyen en formato Apache Parquet, un archivo
por mes:
    https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet

Uso:
    python scripts/download_data.py --year 2023 --months 1 2 3 4 5 6
    python scripts/download_data.py --year 2023            # los 12 meses

Los archivos se guardan en data/raw/ (ignorado por git). Cada archivo mensual
contiene ~3 millones de viajes, de modo que 6-12 meses reunen 20-40M de registros,
volumen suficiente para ejercitar el procesamiento distribuido con Spark.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download_month(year: int, month: int, dest_dir: Path) -> Path:
    """Descarga un archivo mensual si aun no existe. Devuelve la ruta local."""
    fname = f"yellow_tripdata_{year}-{month:02d}.parquet"
    url = f"{BASE_URL}/{fname}"
    dest = dest_dir / fname
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {fname} ya existe ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    print(f"[get ] {url}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"[ok  ] {fname} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Descarga datos NYC TLC Yellow Taxi")
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--months", type=int, nargs="*", default=list(range(1, 7)))
    p.add_argument("--dest", type=Path, default=RAW_DIR)
    args = p.parse_args(argv)

    paths = [download_month(args.year, m, args.dest) for m in args.months]
    total = sum(pth.stat().st_size for pth in paths)
    print(f"\nDescargados {len(paths)} archivos en {args.dest} "
          f"({total/1e6:.1f} MB totales)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
