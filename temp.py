import sqlite3
import pandas as pd  # only if test isn't already loaded in this session
#data wangling
DATA_PATH = "/Users/Skyler/Downloads/hour.csv"
df = pd.read_csv(DATA_PATH)
df['dteday'] = pd.to_datetime(df['dteday'])
df.drop(['casual', 'registered','yr'], axis = 1, inplace = True) #casual and registered are splits of count: perfect predictions,
test = df[(df['dteday'] >= '2011-10-01') & (df['dteday'] <= '2011-12-31')]
testtruth = test[['cnt']]
test = test.drop(['instant', 'dteday','cnt'], axis = 1) #monotonic increases
# from SQLite
with sqlite3.connect('log.db') as conn:
    logged_row = pd.read_sql_query("SELECT * FROM predictions WHERE rowid = 1", conn)
serve_df = df[(df['dteday'] >= '2011-10-01') & (df['dteday'] <= '2011-12-31')]
y_serve = serve_df['cnt']

with sqlite3.connect('log.db') as conn:
    preds = pd.read_sql_query("SELECT prediction FROM predictions WHERE rowid BETWEEN 1 AND 170 ORDER BY rowid", conn)['prediction'].values

true_vals = y_serve.iloc[0:170].values
mae = abs(preds - true_vals).mean()
print(mae)