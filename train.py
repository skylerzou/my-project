import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt #data wrangling

DATA_PATH = "/Users/Skyler/Downloads/hour.csv"
df = pd.read_csv(DATA_PATH)
df['dteday'] = pd.to_datetime(df['dteday'])
df.drop(['casual', 'registered','yr'], axis = 1, inplace = True) #casual and registered are splits of count: perfect predictions,


traindf = df[(df['dteday'] >= '2011-06-01') & (df['dteday'] < '2011-10-01')] #273 is oct 1, 152 is june 1
truthdf= traindf[['cnt']]
traindf = traindf.drop(['instant', 'dteday','cnt'], axis = 1) #monotonic increases


test = df[(df['dteday'] >= '2011-10-01') & (df['dteday'] < '2011-12-31')]
testtruth = test[['cnt']]
test = test.drop(['instant', 'dteday','cnt'], axis = 1) #monotonic increases


#training and saving the model)
model = XGBRegressor(n_estimators= 18, learning_rate=0.1, random_state=42)
model.fit(traindf, truthdf)

preds = model.predict(test)
rmse = np.sqrt(mean_squared_error(testtruth, preds))
r2 = r2_score(testtruth, preds)

#model.save_model('projectboost1.json')
