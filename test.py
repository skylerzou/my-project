import pandas as pd
import torch
import torch.nn as nn


DATA_PATH = "/Users/Skyler/Downloads/day.csv"
df = pd.read_csv(DATA_PATH)
truthdf= df[['cnt']]
df.drop(['casual', 'registered','yr','cnt'], axis = 1, inplace = True) #casual and registered are splits of count: perfect predictions,
traindf = df[(df['instant'] >= 152) & (df['instant'] < 273)] #273 is oct 1, 152 is june 1
df.drop(['instant', 'dteday'], axis = 1, inplace = True) #monotonic increase poison

encoded_df = pd.get_dummies(
    df,
    columns=['season', 'mnth', 'weathersit', 'weekday'],
    drop_first=True
)
print(encoded_df)