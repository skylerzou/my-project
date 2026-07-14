import sqlite3
import pandas as pd
from monitor import drift_check,load_window
import datetime
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


create_table = """CREATE TABLE IF NOT EXISTS model_status (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  status TEXT,
  triggering_window_start INTEGER,
  triggering_window_end INTEGER,
  updated_at TEXT
)"""
seed_row = """INSERT OR IGNORE INTO model_status (id, status, triggering_window_start, triggering_window_end, updated_at)
VALUES (1, 'fresh', NULL, NULL, NULL)"""

with sqlite3.connect('log.db') as conn:
        cursor = conn.cursor()
        cursor.execute(create_table)   
        cursor.execute(seed_row)

#windows
w_start = 1
points = []
while w_start <= 2203:
  points.append((w_start, min(w_start+169, 2203)))
  w_start += 170

#reconstructing the original df
DATA_PATH = "/Users/Skyler/Downloads/hour.csv"
df = pd.read_csv(DATA_PATH)
df['dteday'] = pd.to_datetime(df['dteday'])
df.drop(['casual', 'registered','yr'], axis = 1, inplace = True) #casual and registered are splits of count: perfect predictions,
traindf = df[(df['dteday'] >= '2011-06-01') & (df['dteday'] < '2011-10-01')] #273 is oct 1, 152 is june 1
traindf = traindf.drop(['instant', 'dteday','cnt'], axis = 1) #monotonic increases


EXCLUDE_FROM_QUORUM = {'season', 'mnth'}
for start, end in points:
    results = drift_check(start, end, traindf)
    is_stale = any(r['drifted'] for r in results if r['feature'] not in EXCLUDE_FROM_QUORUM)
    #print(start, end, [r['feature'] for r in results if r['drifted']], is_stale)
    temp_stat = next(r['statistic_value'] for r in results if r['feature'] == 'temp')
    atemp_stat = next(r['statistic_value'] for r in results if r['feature'] == 'atemp')
    print(start, end, f"temp_D={temp_stat:.4f}", f"atemp_D={atemp_stat:.4f}", is_stale)
    if is_stale:
        with sqlite3.connect('log.db') as conn:
          cursor = conn.cursor()
          query = """SELECT status FROM model_status WHERE id=1"""
          cursor.execute(query)
          result = cursor.fetchone()
          current_status = result[0]
          if current_status == 'fresh':
              logger.warning(f"Model marked stale: window {start}-{end}, drifted features: {[r['feature'] for r in results if r['drifted']]}")
              update = """UPDATE model_status SET status = ?, triggering_window_start = ?, triggering_window_end = ?, updated_at = ? WHERE id = 1"""
              updated_at_value = datetime.datetime.now().isoformat()
              cursor.execute(update, ('stale', start, end, updated_at_value))
              conn.commit()
