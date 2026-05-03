import pandas as pd
from bsts import BSTS

OUT_PATH = "../dashboard/src/data/transf.xlsx"
CATEGORIES = ["shake_intensity", "sewer_and_water", "power", "roads_and_bridges", "medical", "buildings"]

# ── 1. Load & pre-process ────────────────────────────────────────────────────
# Data is loaded and time is normalized to 5 minute bins

print("Stage 1/4 — Loading and pre-processing data...")

data = pd.read_csv("data.csv")
data["time"] = pd.to_datetime(data["time"])
data["time_bin"] = data["time"].dt.floor("5min")

LOCATIONS  = sorted(data["location"].unique())


# ── 2. Aggregate (location, category, 5-min bin) ────────────
# Summarize all equal pairs (location, category) to a single value (mean) -> MAP

print("Stage 2/4 — Aggregating to 5-min bins...")

category_dataframes = []
for category in CATEGORIES:
    mean_per_bin = (data.groupby(["location", "time_bin"])[category].mean().reset_index().rename(columns={category: "value"}))
    mean_per_bin = mean_per_bin.dropna(subset=["value"])
    mean_per_bin["category"] = category
    category_dataframes.append(mean_per_bin)

mean_data = pd.concat(category_dataframes, ignore_index=True)


# ── 3. Run BSTS across all (location, category) pairs ────────────────
# Run BSTS model to obtain the data reliability -> CIR

print("Stage 3/4 — Fitting BSTS models on pairs...")

location_category_pairs = list(mean_data.groupby(["location", "category"]))

map_cir_data = []

for i, ((location, category), location_category_data) in enumerate(location_category_pairs, 1):
    location_category_map_cir = BSTS(location, category, location_category_data[["time_bin", "value"]])

    if location_category_map_cir is not None:
        map_cir_data.append(location_category_map_cir)

# ── 4. Combine & export ──────────────────────────────────────────────────────
# Export the results to Excel

print("Stage 4/4 — Combining results and exporting...")

transf_data = pd.concat(map_cir_data, ignore_index=True)
transf_data = transf_data.sort_values(["location", "category", "time_bin"]).reset_index(drop=True)
transf_data["time_bin"] = transf_data["time_bin"].dt.strftime("%Y-%m-%d %H:%M:%S")
transf_data.to_excel(OUT_PATH, index=False, engine="openpyxl")

print(f"Saved to {OUT_PATH}")

