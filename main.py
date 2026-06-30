from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBRegressor
import numpy as np
import sqlite3
from datetime import datetime


app = FastAPI()
featureCols = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
model = XGBRegressor()
model.load_model("projectboost1.json")


database = 'log.db'

create_table = """CREATE TABLE IF NOT EXISTS predictions (
    season INT,
    mnth INT,
    hr INT,
    holiday INT,
    weekday INT,
    workingday INT,
    weathersit INT,
    temp REAL,
    atemp REAL,
    hum REAL,
    windspeed REAL,
    timestamp TEXT,
    prediction REAL
)"""

try:
    with sqlite3.connect('log.db') as conn:
        cursor = conn.cursor()
        cursor.execute(create_table)   
        conn.commit()

except sqlite3.OperationalError as e:
    print(e)


class MyFirstModel(BaseModel):
    season: int
    mnth: int
    hr: int
    holiday: int
    weekday: int
    workingday: int
    weathersit: int
    temp:float
    atemp:float
    hum:float
    windspeed:float



@app.post("/predict")
def predict(data: MyFirstModel):
    raw = data.model_dump()
    ordered = [raw[item] for item in featureCols]
    result = model.predict(np.array([ordered]))
    now = datetime.now().isoformat()
    with sqlite3.connect("log.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""INSERT INTO predictions (season, mnth, hr, holiday ,weekday ,workingday ,weathersit, temp ,atemp, hum ,windspeed ,timestamp , prediction ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
    (tuple(ordered + [now] + [float(result[0])]))
)
        conn.commit()


    return {"prediction": float(result[0])}