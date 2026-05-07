import pandas as pd
from multiprocessing import Pool, cpu_count

from bsts import run_pair

OUT_PATH   = "../dashboard/src/data/transf.xlsx"
CATEGORIES = ["shake_intensity", "sewer_and_water", "power", "roads_and_bridges", "medical", "buildings"]

if __name__ == "__main__":

    # ── 1. Load & pre-process ────────────────────────────────────────────────
    print("Stage 1/4 — Loading and pre-processing data...")

    data = pd.read_csv("data.csv")
    data["time"] = pd.to_datetime(data["time"])
    data["time_bin"] = data["time"].dt.floor("5min")  # agrupar en bins de 5 min como el R

    # ── 2. Melt a formato largo — una fila por reporte por categoría ─────────
    print("Stage 2/4 — Reshaping to long format...")

    data = data.melt(id_vars=["time_bin", "location"], value_vars=CATEGORIES, var_name="category", value_name="value").dropna(subset=["value"])

    # ── 3. Agrupar y correr BSTS por par (location, category) ────────────────
    pairs = [((location, category), group) for (location, category), group in data.groupby(["location", "category"]) if len(group) >= 5]

    n_workers = max(1, cpu_count() - 1)
    print(f"Stage 3/4 — Fitting BSTS models on {len(pairs)} pairs using {n_workers} workers...")

    with Pool(processes=n_workers) as pool:
        results = pool.map(run_pair, pairs)

    map_cir_data = [r for r in results if r is not None]

    # ── 4. Combine & export ──────────────────────────────────────────────────
    print("Stage 4/4 — Combining results and exporting...")

    transf_data = pd.concat(map_cir_data, ignore_index=True)
    transf_data = transf_data.sort_values(["location", "category", "time_bin"]).reset_index(drop=True)
    transf_data["time_bin"] = transf_data["time_bin"].astype(str)
    transf_data.to_excel(OUT_PATH, index=False, engine="openpyxl")

    print(f"Saved to {OUT_PATH}")