import numpy as np
from scipy import stats

def ks (reference_array, batch_array):
    if len(reference_array) >= 30 and len(batch_array) >= 30:
        ksobj = stats.ks_2samp(reference_array, batch_array, alternative='two-sided')
        ksres = ksobj.statistic
        return {"statistic_name": "KS", "statistic_value": ksres}
    raise ValueError(f"Arrays must have at least 30 elements: reference has {len(reference_array)}, batch has {len(batch_array)}")

print(ks(np.random.uniform(0, 1, 35), np.random.uniform(10, 11, 35)))