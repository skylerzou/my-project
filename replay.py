import pandas as pd
import requests
import sqlite3

#data wangling
DATA_PATH = "/Users/Skyler/Downloads/hour.csv"
df = pd.read_csv(DATA_PATH)
df['dteday'] = pd.to_datetime(df['dteday'])
df.drop(['casual', 'registered','yr'], axis = 1, inplace = True) #casual and registered are splits of count: perfect predictions,
test = df[(df['dteday'] >= '2011-10-01') & (df['dteday'] <= '2011-12-31')]
testtruth = test[['cnt']]
test = test.drop(['instant', 'dteday','cnt'], axis = 1) #monotonic increases

session = requests.Session()
log = [] 

with sqlite3.connect('log.db') as conn:
            cursor = conn.cursor()
            cursor.execute("""SELECT COUNT(*) FROM predictions""")   
            row = cursor.fetchone()
            precount = row[0]

for i in range(len(test)):
    one_row_df = test.iloc[[i]]
    payload = one_row_df.to_dict('records')[0]
    try:
        response = session.post("http://localhost:8000/predict", json=payload)
        status = response.status_code
        if status != 200:
              #log it
              #continue
              log.append({"row": i, "status": status, "predicted": None, "error": f"HTTP {status}"})
              continue
            
        package = response.json()
        predicted = package["prediction"]
        log.append({"row": i, "status": status, "predicted": predicted, "error": None})
    except requests.exceptions.RequestException as e:
        log.append({"row": i, "status": None, "predicted": None, "error": str(e)})
        continue
          
with sqlite3.connect('log.db') as conn:
    cursor = conn.cursor()
    cursor.execute("""SELECT COUNT(*) FROM predictions""")   
    row = cursor.fetchone()
    postcount = row[0]

successes = sum(1 for entry in log if entry["status"] == 200)
assert postcount - precount == successes, f"DB grew by {postcount-precount}, expected {successes}"
assert len(log) == len(test), f"logged {len(log)} entries, expected {len(test)}"
pd.DataFrame(log).to_csv("log.csv", index=False)

print(precount, postcount, postcount - precount, successes, len(log))
