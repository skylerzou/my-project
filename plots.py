import sqlite3
import pandas as pd  # only if test isn't already loaded in this session
from monitor import drift_check
import matplotlib.pyplot as plt
from monitor import KS_THRESHOLD
#data wangling
DATA_PATH = "/Users/Skyler/Downloads/hour.csv"
df = pd.read_csv(DATA_PATH)
df['dteday'] = pd.to_datetime(df['dteday'])
df.drop(['casual', 'registered','yr'], axis = 1, inplace = True) #casual and registered are splits of count: perfect predictions,
test = df[(df['dteday'] >= '2011-10-01') & (df['dteday'] <= '2011-12-31')]
testtruth = test['cnt']
test = test.drop(['instant', 'dteday','cnt'], axis = 1) #monotonic increases


traindf = df[(df['dteday'] >= '2011-06-01') & (df['dteday'] < '2011-10-01')]
traindf = traindf.drop(['instant', 'dteday','cnt'], axis = 1)


print(test.index[0] == testtruth.index[0])
# from SQLite
error_series = []
drift_series = []
w_start = 1
points = []
while w_start <= 2203:
  points.append((w_start, min(w_start+169, 2203)))
  w_start += 170

for start, end in points: 
    results = drift_check(start, end, traindf)
    temp_stat = next(r['statistic_value'] for r in results if r['feature'] == 'temp')

    with sqlite3.connect('log.db') as conn:
        preds = pd.read_sql_query(
            "SELECT prediction FROM predictions WHERE rowid BETWEEN ? AND ? ORDER BY rowid",
            conn, params=(start, end)
        )['prediction'].values

    true_vals =testtruth.iloc[start-1:end].values
    mae = abs(preds - true_vals).mean()

    error_series.append(mae)
    drift_series.append(temp_stat)

fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
windows = list(range(1, 14))

# Top panel — drift
ax_top.plot(windows, drift_series, marker='o')
ax_top.axhline(KS_THRESHOLD, linestyle='--', color='gray', label='calibrated threshold')
ax_top.set_ylim(0, 1)
ax_top.set_ylabel("temp KS D statistic")
ax_top.set_title("The statistic saturates near w1")
ax_top.legend()

# Bottom panel — error
ax_bottom.plot(windows, error_series, marker='o', color='tab:red')
ax_bottom.set_ylim(bottom=0)
ax_bottom.set_ylabel("MAE (rentals/hour)")
ax_bottom.set_xlabel("serve window (~170 rows ≈ 1 week)")
ax_bottom.set_title("Error keeps climbing after the statistic pins")

ax_bottom.set_xlim(0.5, 13.5)

# Killer pair shading
for ax in (ax_top, ax_bottom):
    ax.axvspan(9.5, 10.5, alpha=0.15, color='orange')
    ax.axvspan(12.5, 13.5, alpha=0.15, color='orange')

ax_top.annotate("w10 vs w13: ΔD ≈ 0.004, MAE 2.05×",
                xy=(11.5, 0.5), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))

fig.suptitle("KS D-statistic is bounded; the covariate shift and its cost are not")
plt.tight_layout()
plt.savefig('drift_vs_error.png', dpi=150)

psi_series = []
for start, end in points:
    results = drift_check(start, end, traindf)
    psi_val = next(r['statistic_value'] for r in results if r['feature'] == 'holiday')
    psi_series.append(psi_val)
    print(start, end, f"PSI={psi_val:.4f}")