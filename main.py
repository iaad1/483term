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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_log_error, median_absolute_error, explained_variance_score, mean_squared_error


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
#so we need to resize it
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

#inverse transform the scaled predictions
#rescalling data so that metrics analysis makes sense
valPredictions = targetScaler.inverse_transform(valPredictionsScaled.reshape(-1, 1))

yValOriginal = targetScaler.inverse_transform(yVal.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(yValOriginal, valPredictions))
mae = mean_absolute_error(yValOriginal, valPredictions)
r2 = r2_score(yValOriginal, valPredictions)
#msle = mean_squared_log_error(yValOriginal, valPredictions)
median_ae = median_absolute_error(yValOriginal, valPredictions)
explained_variance = explained_variance_score(yValOriginal, valPredictions)

print(f'Validation RMSE: {rmse}')
print(f'Validation MAE: {mae}')
print(f'Validation R-squared: {r2}')
#print(f'Validation MSLE: {msle}')
print(f'Validation Median Absolute Error: {median_ae}')
print(f'Validation Explained Variance Score: {explained_variance}')