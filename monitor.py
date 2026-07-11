import numpy as np
from scipy import stats
import pandas as pd
import sqlite3

def ks (reference_array, batch_array):
   # if len(reference_array) >= 30 and len(batch_array) >= 30:
    ksobj = stats.ks_2samp(reference_array, batch_array, alternative='two-sided')
    ksres = ksobj.statistic
    return {"statistic_name": "KS", "statistic_value": ksres}
    #raise ValueError(f"Arrays must have at least 30 elements: reference has {len(reference_array)}, batch has {len(batch_array)}")

def psi(reference_array, batch_array, numBins):
    expected_counts= np.bincount(reference_array, minlength=numBins)
    actual_counts = np.bincount(batch_array, minlength=numBins)
    #smoothing
    expected_counts += 1
    actual_counts += 1
    #proportions
    expected = expected_counts / expected_counts.sum()
    actual = actual_counts / actual_counts.sum()

    psires = ((actual - expected) * np.log(actual/ expected)).sum()

    return {"statistic_name": "PSI", "statistic_value": psires}

def load_window(rowid_start, rowid_end):
    with sqlite3.connect('log.db') as conn:
        return pd.read_sql_query(
            """SELECT * FROM predictions WHERE rowid BETWEEN ? AND ?""",
            conn,
            params=(rowid_start, rowid_end))
    
#Note: copied from training any changes need to be shared between the two
DATA_PATH = "/Users/Skyler/Downloads/hour.csv"
df = pd.read_csv(DATA_PATH)
df['dteday'] = pd.to_datetime(df['dteday'])
df.drop(['casual', 'registered','yr'], axis = 1, inplace = True) #casual and registered are splits of count: perfect predictions,
traindf = df[(df['dteday'] >= '2011-06-01') & (df['dteday'] < '2011-10-01')] #273 is oct 1, 152 is june 1
traindf = traindf.drop(['instant', 'dteday','cnt'], axis = 1) #monotonic increases


PSI_THRESHOLD = 0.2
KS_THRESHOLD = 0.15

def drift_check(rowid_start, rowid_end, original):
    results = []
    window_df = load_window(rowid_start, rowid_end)
    CONTINUOUS = ['temp', 'atemp', 'hum', 'windspeed']
    CATEGORICAL = {'season': 4, 'mnth': 12, 'hr': 24, 'weekday': 7, 'weathersit': 4, 'holiday': 2, 'workingday': 2}
    for feat in CONTINUOUS:
        r = ks(original[feat].values, window_df[feat].values)
        r['drifted'] = r['statistic_value'] > KS_THRESHOLD
        r['feature'] = feat
        results.append(r)
    for feat, n_bins in CATEGORICAL.items():
        shift = 1 if feat in ('season','mnth','weathersit') else 0
        r = psi(original[feat].values - shift, window_df[feat].values - shift, n_bins)
        r['drifted'] = r['statistic_value'] > PSI_THRESHOLD
        r['feature'] = feat
        results.append(r)
    return results

#testing edge cases

#PSI NULL CASE
#np.random.seed(42)
# probs = [0.25, 0.25, 0.25, 0.25]
# categories = [0, 1, 2, 3]
# reference = np.random.choice(categories, size=1000, p=probs)
# batch = np.random.choice(categories, size=300, p=probs)
# psi_value = psi(reference, batch, 4)
# psi_value = psi_value['statistic_value']
# assert psi_value < 0.1, f"expected no-drift band (<0.1), got {psi_value}"
# print(f"PSI null case: {psi_value:.4f} — PASS")


#KS NULL CASE
# reference = np.random.normal(loc=0.5, scale=0.15, size=1000)
# batch = np.random.normal(loc=0.5, scale=0.15, size=300)  # same params — null
# ks_result = ks(reference, batch)
# d_null = ks_result['statistic_value']
# print(f"KS null D-statistic: {d_null:.4f}")


# reference = np.random.normal(loc=0.5, scale=0.15, size=1000)
# batch = np.random.normal(loc=0.15, scale=0.15, size=300)  # ~0.35 shift — matches Day 1 oracle magnitude

# ks_result = ks(reference, batch)
# d_drift = ks_result['statistic_value']

# print(f"KS drift D-statistic: {d_drift:.4f}")

#PSI MISSING CATEGORY
# probs = [0.25, 0.25, 0.25, 0.25]
# probsmissing = [0, 0.5, 0.25, 0.25]
# categories = [0, 1, 2, 3]
# reference = np.random.choice(categories, size=1000, p=probs)
# batch = np.random.choice(categories, size=300, p=probsmissing)
# psival = psi(reference, batch, 4)
# psi_value = psival['statistic_value']

#PSI tiny batch
# probs = [0.25, 0.25, 0.25, 0.25]
# categories = [0, 1, 2, 3]
# reference = np.random.choice(categories, size=1000, p=probs)
# batch = np.random.choice(categories, size=3, p=probs)
# psival = psi(reference, batch, 4)
# psi_value = psival['statistic_value']
# print(psi_value)
#RESULT: PSI UNRELIABLE BELOW MINMUM BATCH SIZE, NEED MINIMUM N

#KS tiny batch
# reference = np.random.normal(loc=0.5, scale=0.15, size=1000)
# batch = np.random.normal(loc=0.5, scale=0.15, size=3)  # same params — null
# ks_result = ks(reference, batch)
# d = ks_result['statistic_value']
# print(d)

#KS zero variance batch
# reference = np.random.normal(loc=0.5, scale=0.15, size=1000)

# # Variant A — degenerate batch AT reference's median
# batch_a = np.full(10, 0.5)
# d_a = ks(reference, batch_a)['statistic_value']
# print(f"A (point mass at median): D={d_a:.4f}")

# # Variant B — degenerate batch FAR OUTSIDE reference's range
# batch_b = np.full(10, 2.0)
# d_b = ks(reference, batch_b)['statistic_value']
# print(f"B (point mass far outside range): D={d_b:.4f}")

# assert np.isfinite(d_a) and np.isfinite(d_b)

#PSI zero variance
# np.random.seed(42)
# categories = [0, 1, 2, 3]
# reference_probs = [0.25, 0.25, 0.25, 0.25]
# reference = np.random.choice(categories, size=1000, p=reference_probs)
# batch = np.full(50, 2)  # every single sample is category 2

# psi_val = psi(reference, batch, 4)['statistic_value']
# print(f"PSI all-one-category: {psi_val:.4f}")

np.random.seed(42)
reference = np.full(1000, 0.5)  # zero-variance reference

# Sub-case 1: batch matches reference's constant value
batch_same = np.full(10, 0.5)
d_same = ks(reference, batch_same)['statistic_value']
print(f"Reference degenerate, batch matches: D={d_same:.4f}")

# Sub-case 2: batch is a different constant
batch_diff = np.full(10, 2.0)
d_diff = ks(reference, batch_diff)['statistic_value']
print(f"Reference degenerate, batch differs: D={d_diff:.4f}")