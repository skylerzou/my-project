import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


DATA_PATH = "/Users/Skyler/Downloads/day.csv"
df = pd.read_csv(DATA_PATH)
df.drop(['casual', 'registered','yr'], axis = 1, inplace = True) #casual and registered are splits of count: perfect predictions,
df = pd.get_dummies(df, columns=['season', 'mnth', 'weathersit', 'weekday'], drop_first=True)


traindf = df[(df['instant'] >= 152) & (df['instant'] < 273)] #273 is oct 1, 152 is june 1
truthdf= traindf[['cnt']]
traindf.drop(['instant', 'dteday','cnt'], axis = 1, inplace = True) #monotonic increases


test = df[(df['instant'] >= 274) & (df['instant'] < 365)]
testtruth = test[['cnt']]
test.drop(['instant', 'dteday','cnt'], axis = 1, inplace = True) #monotonic increases



model = XGBRegressor(n_estimators= 18, learning_rate=0.1, random_state=42)
model.fit(traindf, truthdf)

preds = model.predict(test)
rmse = np.sqrt(mean_squared_error(testtruth, preds))
r2 = r2_score(testtruth, preds)

