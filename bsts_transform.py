import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents
from joblib import Parallel, delayed
import time as _time

# ── 1. Load & pre-process ────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv("mc1-reports-data.csv")
df["time"] = pd.to_datetime(df["time"])

df["time_bin"] = df["time"].dt.floor("5min")

CATEGORIES = ["shake_intensity", "sewer_and_water", "power",
              "roads_and_bridges", "medical", "buildings"]
LOCATIONS  = sorted(df["location"].unique())

# ── 2. Aggregate: mean rating per (location, category, 5-min bin) ────────────

print("Aggregating to 5-min bins...")
records = []
for cat in CATEGORIES:
    agg = (
        df.groupby(["location", "time_bin"])[cat]
        .mean()
        .reset_index()
        .rename(columns={cat: "value"})
    )
    agg = agg.dropna(subset=["value"])
    agg["category"] = cat
    records.append(agg)

agg_df = pd.concat(records, ignore_index=True)

# ── 3. BSTS local-level model per (location, category) ─────────────────────

def run_bsts(location, category, series_df):

    s = (series_df
         .set_index("time_bin")["value"]
         .sort_index()
         .dropna())

    n = len(s)
    if n < 5:
        return None

    y = s.values.astype(float)

    prior_mean = float(y[0])
    prior_std  = 0.1 * float(np.std(y)) if np.std(y) > 0 else 1e-4
    prior_var  = prior_std ** 2

    try:
        model = UnobservedComponents(
            y,
            level="local level",
        )
        res = model.fit(
            initialization="known",
            initial_state=np.array([prior_mean]),
            initial_state_cov=np.array([[prior_var]]),
            method="lbfgs",
            disp=False,
            maxiter=200,
        )
    except Exception:
        return None

    filt_mean = res.filtered_state[0]
    filt_var  = res.filtered_state_cov[0, 0]
    filt_sd   = np.sqrt(np.maximum(filt_var, 0))

    z = {"50": 0.6745, "80": 1.2816, "95": 1.9600}

    rows = []
    for i, t in enumerate(s.index):
        mu = filt_mean[i]
        sd = filt_sd[i]
        row = {
            "time_bin":  t,
            "location":  location,
            "category":  category,
            "map":       round(float(mu), 4),
        }
        for pct, zval in z.items():
            lo = float(mu - zval * sd)
            hi = float(mu + zval * sd)
            row[f"ci{pct}_lo"] = round(lo, 4)
            row[f"ci{pct}_hi"] = round(hi, 4)

        lo95_clipped = max(row["ci95_lo"], 0.0)
        hi95_clipped = min(row["ci95_hi"], 10.0)
        row["cir"] = round(max(hi95_clipped - lo95_clipped, 0.0), 4)

        rows.append(row)

    return pd.DataFrame(rows)


# ── 4. Run in parallel across all (location, category) pairs ────────────────

pairs = list(agg_df.groupby(["location", "category"]))
print(f"\nRunning BSTS on {len(pairs)} (location × category) pairs...")

t0 = _time.time()

results = []
for i, ((loc, cat), grp) in enumerate(pairs, 1):
    out = run_bsts(loc, cat, grp[["time_bin", "value"]])
    if out is not None:
        results.append(out)
    if i % 20 == 0:
        elapsed = _time.time() - t0

# ── 5. Combine & export ──────────────────────────────────────────────────────

final = pd.concat(results, ignore_index=True)
final = final.sort_values(["location", "category", "time_bin"]).reset_index(drop=True)

final["time_bin"] = final["time_bin"].dt.strftime("%Y-%m-%d %H:%M:%S")
out_path = "mc1_bsts_map_cir_py.xlsx"
final.to_excel(out_path, index=False, engine="openpyxl")

print(f"\n✓ Saved {len(final):,} rows → {out_path}")
print(final.head(10).to_string())
print("\nColumn stats:")
print(final[["map", "ci95_lo", "ci95_hi", "cir"]].describe().round(3))
