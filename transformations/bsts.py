# bsts.py
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import gaussian_kde


def BSTS(location, category, location_category_data):

    time_series = (location_category_data.set_index("time_bin")["value"].sort_index().dropna())

    if len(time_series) < 5:
        return None

    HDI_LEVELS = [0.50, 0.80, 0.90, 0.95]
    NKEEP = 200
    BURN  = 100

    ratings   = time_series.values.astype(float)
    time_bins = time_series.index
    n         = len(ratings)

    def hdi(samples, prob):
        sorted_s   = np.sort(samples)
        n_s        = len(sorted_s)
        n_included = int(np.floor(prob * n_s))
        widths     = sorted_s[n_included:] - sorted_s[:n_s - n_included]
        idx        = int(np.argmin(widths))
        return float(sorted_s[idx]), float(sorted_s[idx + n_included])

    output_rows = []

    for t in range(n):
        y_obs = ratings[: t + 1]
        T     = len(y_obs)

        with pm.Model():
            sigma_level = pm.HalfNormal("sigma_level", sigma=1.0)

            mu0 = pm.Normal("mu0", mu=np.log(np.clip(y_obs[0], 0.1, None)), sigma=1.0)

            if T > 1:
                innovations = pm.Normal("innovations", mu=0, sigma=sigma_level, shape=T - 1)
                log_mu = pm.Deterministic("log_mu", pm.math.concatenate([mu0[None], mu0 + pm.math.cumsum(innovations)]))
            else:
                log_mu = pm.Deterministic("log_mu", mu0[None])

            pm.Poisson("y", mu=pm.math.exp(log_mu), observed=y_obs)

            trace = pm.sample(draws=NKEEP, tune=BURN, chains=1, progressbar=False, random_seed=0, target_accept=0.9, return_inferencedata=True)

        posterior_last = np.exp(trace.posterior["log_mu"].values[0, :, -1])

        kde     = gaussian_kde(posterior_last)
        grid    = np.linspace(posterior_last.min(), posterior_last.max(), 1000)
        map_val = float(grid[np.argmax(kde(grid))])

        row = {"time_bin": time_bins[t], "location": location, "category": category,
                "map": round(map_val,4), "mean": round(float(posterior_last.mean()), 4)}

        for prob in HDI_LEVELS:
            lo, hi = hdi(posterior_last, prob)
            key = str(int(prob * 100))
            row[f"ci{key}_lo"] = round(lo, 4)
            row[f"ci{key}_hi"] = round(hi, 4)

        hi95, lo95 = row["ci95_hi"], row["ci95_lo"]
        row["cir"] = round((10 - lo95) if hi95 > 10 else (hi95 - lo95), 4)

        output_rows.append(row)

    return pd.DataFrame(output_rows)


def run_pair(args):
    (location, category), group = args
    return BSTS(location, category, group[["time_bin", "value"]])