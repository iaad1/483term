# shbang
# Judah Tanninen
# Shawyan Tabari

# Term 483 Project
# Kaggle: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

# Imports
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Turn all the csvs into dataframes using pandas.
holidayDf = pd.read_csv('holiday_events.csv')
oilDf = pd.read_csv('oil.csv')
trainDf = pd.read_csv('train.csv')
testDf = pd.read_csv('test.csv')