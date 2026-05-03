import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

def BSTS(location, category, location_category_data):

    time_series = (location_category_data.set_index("time_bin")["value"].sort_index().dropna())

    num_observations = len(time_series)
    
    if num_observations < 5:
        return None

    ratings = time_series.values.astype(float)

    prior_mean = float(ratings[0])
    prior_std  = 0.1 * float(np.std(ratings)) if np.std(ratings) > 0 else 1e-4
    prior_var  = prior_std ** 2

    try:
        model = UnobservedComponents(ratings, level="local level")
        model.initialize_known(initial_state=np.array([prior_mean]), initial_state_cov=np.array([[prior_var]]))
        fitted_model = model.fit(method="lbfgs", disp=False, maxiter=200,)
    except:
        return None

    filtered_mean = fitted_model.filtered_state[0]
    filtered_var  = fitted_model.filtered_state_cov[0, 0]
    filtered_std  = np.sqrt(np.maximum(filtered_var, 0))

    confidence_intervals = {"50": 0.6745, "80": 1.2816, "95": 1.9600}

    output_rows = []
    for i, time_bin in enumerate(time_series.index):
        estimated_mean = filtered_mean[i]
        estimated_std  = filtered_std[i]
        row = {"time_bin": time_bin, "location": location, "category": category, "map": round(float(estimated_mean), 4)}
        
        for confidence_level, z_score in confidence_intervals.items():
            lower_bound = float(estimated_mean - z_score * estimated_std)
            upper_bound = float(estimated_mean + z_score * estimated_std)
            row[f"ci{confidence_level}_lo"] = round(lower_bound, 4)
            row[f"ci{confidence_level}_hi"] = round(upper_bound, 4)

        lower_bound_95_clipped = max(row["ci95_lo"], 0.0)
        upper_bound_95_clipped = min(row["ci95_hi"], 10.0)
        row["cir"] = round(max(upper_bound_95_clipped - lower_bound_95_clipped, 0.0), 4)

        output_rows.append(row)

    return pd.DataFrame(output_rows)