"""
Matriculacions mensuals de vehicles electrificats/híbrids a Catalunya
Font: CSV Generalitat de Catalunya
Exporta: total_matriculacions_mensual.csv (totes les tecnologies agregades)
"""

import csv
import os
from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_CSV  = os.path.join(PROJECT_DIR, "data", "raw", "Matriculacions_mensuals_de_vehicles_electrificats_i_híbrids_a_Catalunya_20260525.csv")
OUTPUT_CSV = os.path.join(PROJECT_DIR, "data", "processed", "total_matriculacions_mensual.csv")


def load_and_aggregate(filepath: str) -> list[dict]:
    totals = defaultdict(int)
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mes = int(row["Mes"])
                any_ = int(row["Any"])
                uts = int(row["Uts"])
            except (ValueError, KeyError):
                continue
            key = f"{any_:04d}-{mes:02d}"
            totals[key] += uts

    return [{"any_mes": k, "total_matriculacions": v}
            for k, v in sorted(totals.items())]


def save_csv(data: list[dict], filepath: str):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["any_mes", "total_matriculacions"])
        writer.writeheader()
        writer.writerows(data)
    print(f"  ✓ Desat: {filepath} ({len(data)} mesos)")


if __name__ == "__main__":
    print("=" * 55)
    print("  Matriculacions mensuals — totes les tecnologies")
    print("=" * 55)

    data = load_and_aggregate(INPUT_CSV)
    save_csv(data, OUTPUT_CSV)

    print("\n--- Últims 12 mesos ---")
    print(f"{'Mes':<10} {'Total':>10}")
    print("-" * 22)
    for row in data[-12:]:
        print(f"{row['any_mes']:<10} {row['total_matriculacions']:>10}")

    total = sum(r["total_matriculacions"] for r in data)
    print("\n" + "=" * 55)
    print(f"  TOTAL acumulat: {total:,} matriculacions")
    print(f"  Periode: {data[0]['any_mes']} → {data[-1]['any_mes']}")
    print("=" * 55)
