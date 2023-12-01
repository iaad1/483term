# shbang
# Judah Tanninen
# Shawyan Tabari
# Elesey Razumovskiy

# Term 483 Project
# Kaggle: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

# Imports
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor


# STEP 1 (SECURE THE KEYS!)
# Read all the relevant csvs into the dataframes.
print('READING CSVS')
holidayDf = pd.read_csv('holidays_events.csv')
oilDf = pd.read_csv('oil.csv')
trainDf = pd.read_csv('train.csv')
testDf = pd.read_csv('test.csv')

# STEP 2 (ASCEND FROM DARKNESS!)
# Merge the holiday and oil data onto the training and test frames
trainDf = pd.merge(trainDf, oilDf, on="date", how="left") # Cute little left join
trainDf = pd.merge(trainDf, holidayDf, on="date", how="left")
testDf = pd.merge(testDf, oilDf, on="date", how="left") # Cute little left join
testDf = pd.merge(testDf, holidayDf, on="date", how="left")


# STEP 3 (RAIN FIRE)
# Prepare the dataset, convert values

# Convert onpromotion to a boolean value
trainDf['onpromotion'] = (trainDf['onpromotion'] > 0).astype(int)
testDf['onpromotion'] = (testDf['onpromotion'] > 0).astype(int)

# Change all the holiday junk to a simple is_holiday numerical column.
trainDf['is_holiday'] = (~trainDf['type'].isna()).astype(int)
testDf['is_holiday'] = (~testDf['type'].isna()).astype(int)

# Create a list of the columns that we want
desired = ['store_nbr', 'family', 'sales', 'onpromotion', 'dcoilwtico', 'is_holiday']