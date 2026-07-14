import numpy as np
from scipy import stats
import pandas as pd
import sqlite3
def ks (reference_array, batch_array):
    if len(reference_array) >= 30 and len(batch_array) >= 30:
        ksobj = stats.ks_2samp(reference_array, batch_array, alternative='two-sided')
        ksres = ksobj.statistic
        return {"statistic_name": "KS", "statistic_value": ksres}
    raise ValueError(f"Arrays must have at least 30 elements: reference has {len(reference_array)}, batch has {len(batch_array)}")

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
