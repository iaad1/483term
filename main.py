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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint, uniform
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
import matplotlib.pyplot as plt


# STEP 1 (SECURE THE KEYS!)
# Read all the relevant csvs into the dataframes.
print('READING CSVS')
holidayDf = pd.read_csv('holidays_events.csv')
oilDf = pd.read_csv('oil.csv')
trainDf = pd.read_csv('train.csv')
testDf = pd.read_csv('test.csv')
transactionsDf = pd.read_csv('transactions.csv')

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


#Graphing the data that we have to visualize it
# Convert 'date' to datetime and sort
transactionsDf['date'] = pd.to_datetime(transactionsDf['date'])
transactionsDf.sort_values('date', inplace=True)

# Aggregate transactions by date
daily_transactions = transactionsDf.groupby('date')['transactions'].sum().reset_index()

daily_transactions['7_day_MA'] = daily_transactions['transactions'].rolling(window=7).mean()
daily_transactions['14_day_MA'] = daily_transactions['transactions'].rolling(window=14).mean()
daily_transactions['30_day_MA'] = daily_transactions['transactions'].rolling(window=30).mean()
daily_transactions['60_day_MA'] = daily_transactions['transactions'].rolling(window=60).mean()

plt.figure(figsize=(12, 6))
plt.plot(daily_transactions['date'], daily_transactions['transactions'], label='Total Daily Transactions')
plt.plot(daily_transactions['date'], daily_transactions['7_day_MA'], label='7-Day MA')
plt.plot(daily_transactions['date'], daily_transactions['14_day_MA'], label='14-Day MA')
plt.plot(daily_transactions['date'], daily_transactions['30_day_MA'], label='30-Day MA')
plt.plot(daily_transactions['date'], daily_transactions['60_day_MA'], label='60-Day MA')
plt.title('Total Transactions Over Time with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Transactions')
plt.legend()
plt.show()


#imputers for handling missing values
numImp = SimpleImputer(strategy='mean')
catImp = SimpleImputer(strategy='most_frequent')

#preprocessing pipelines
numPipe = Pipeline(steps=[
    ('imputer', numImp),
    ('scaler', MinMaxScaler())
])

catPipe = Pipeline(steps=[
    ('imputer', catImp),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numPipe, ['onpromotion', 'dcoilwtico', 'is_holiday', 'store_nbr']),
        ('cat', catPipe, ['family'])
    ])

#hyperparas
hyperParam = {
    'model__n_estimators': randint(50, 1000), #made this large since data set is in the millions of rows
    'model__learning_rate': uniform(0.01, 0.3), 
    'model__max_depth': randint(2, 10),  
    'model__min_samples_split': randint(2, 10),  
    'model__min_samples_leaf': randint(1, 10), 
    'model__max_features': ['sqrt', 'log2'],  
    'model__subsample': uniform(0.5, 0.5),  #fraction of samples to be used for fitting the individual base learners
}

#scale the target variable 'sales' to range 0-1, I have to go back to origianl values
#when doing the analysis since it is scaled differently and the analysis will look weird for 0-1
targetScaler = MinMaxScaler()
yTrainScaled = targetScaler.fit_transform(trainDf[['sales']])

#splitting the data for training and validation
XTrain, XVal, yTrain, yVal = train_test_split(
    trainDf.drop('sales', axis=1), 
    yTrainScaled, 
    test_size=0.2, 
    random_state=42
)

#flattening y_train and y_val for model fitting and validation
#gradient boost expects a 1D array but minmaxscaler generates a 2D array
#we mighte need to resize it, can test without 
yTrain = yTrain.ravel()
yVal = yVal.ravel()


#pipeline with GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

#randomizedSearchCV
randomSearch = RandomizedSearchCV(
    pipeline, 
    param_distributions=hyperParam, 
    n_iter=10, 
    cv=5, 
    scoring='neg_mean_squared_error', 
    random_state=42,
    n_jobs=-1
)

#fitting
randomSearch.fit(XTrain, yTrain)
bestModel = randomSearch.best_estimator_

print("Best hyperparameters:")
print(randomSearch.best_params_)


#validation
valPredictionsScaled = bestModel.predict(XVal)

# Ensure that predictions and actual values are greater than 0
yValOriginalClipped = np.clip(yValOriginal, a_min=0.01, a_max=None)  # Replace 0 and negative values with a small positive number
valPredictionsClipped = np.clip(valPredictions, a_min=0.01, a_max=None)

# Calculate metrics
mae = mean_absolute_error(yValOriginalClipped, valPredictionsClipped)
r2 = r2_score(yValOriginalClipped, valPredictionsClipped)
rmsle = np.sqrt(mean_squared_log_error(yValOriginalClipped, valPredictionsClipped))


# Print metrics
print(f'Validation MAE: {mae}')
print(f'Validation R-squared: {r2}')
print(f'Validation RMSLE: {rmsle}')


#visulization
# Plotting Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(yValOriginal, valPredictions, alpha=0.5)
plt.plot(yValOriginal, yValOriginal, color="red")  # Line showing perfect predictions
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()

# Calculating residuals
residuals = yValOriginal - valPredictions

# Plotting residuals
plt.figure(figsize=(10, 6))
plt.scatter(yValOriginal, residuals, alpha=0.5)
plt.hlines(y=0, xmin=yValOriginal.min(), xmax=yValOriginal.max(), colors='red')
plt.title('Residuals of Predictions')
plt.xlabel('Actual Sales')
plt.ylabel('Residuals')
plt.show()


# Final predictions on test data (can be used as needed)
# X_test = test_data.drop('id', axis=1)
# test_data['sales'] = best_model.predict(X_test)
# output = test_data[['id', 'sales']]
# output.to_csv('predictions.csv', index=False)