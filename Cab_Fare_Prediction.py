# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:13:06 2020

@author: hp
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pickle
from geopy.distance import geodesic
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder 
from sklearn import ensemble

# Importing the dataset
dataset = pd.read_csv('train_cab.csv')
dataset.head()
missing_values = dataset.isnull().sum()
missing_percentage = missing_values/len(dataset) * 100 
dataset = dataset.dropna()
dataset.dtypes
dataset.info()
dataset['pickup_datetime']= pd.to_datetime(dataset['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC', errors = 'coerce')

dataset.head()

dataset['pickup_date']= dataset['pickup_datetime'].dt.date
dataset['pickup_day']=dataset['pickup_datetime'].apply(lambda x:x.day)
dataset['pickup_hour']=dataset['pickup_datetime'].apply(lambda x:x.hour)

dataset['pickup_month']=dataset['pickup_datetime'].apply(lambda x:x.month)
dataset['pickup_year']=dataset['pickup_datetime'].apply(lambda x:x.year)

#calculate trip distance in kms
def distance(lat1, lat2, lon1,lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) * 1.609

dataset['trip_distance'] = dataset.apply(lambda row:distance(row['pickup_latitude'],
     row['dropoff_latitude'],row['pickup_longitude'],row['dropoff_longitude']),axis=1)


#Once we have the distance column and day, hour, month and Year columns sourcecd, 
#we delete the columns of latitude and longitude and pickup date time
dataset.drop('pickup_latitude', axis=1, inplace=True)
dataset.drop('dropoff_latitude', axis=1, inplace=True)
dataset.drop('pickup_longitude', axis=1, inplace=True)
dataset.drop('dropoff_longitude', axis=1, inplace=True)
dataset.drop('pickup_datetime', axis=1, inplace=True)
dataset.drop('pickup_date', axis=1, inplace=True)

#Converting the columns to appropriate data types:
dataset['passenger_count'] = dataset['passenger_count'].astype(int)
dataset['fare_amount'] = dataset['fare_amount'].str.rstrip('-').astype('float')
      
dataset['trip_distance'] = dataset['trip_distance'].astype(float)
dataset['pickup_day'] = dataset['pickup_day'].astype(object)
dataset['pickup_hour'] = dataset['pickup_hour'].astype(object)
dataset['pickup_month'] = dataset['pickup_month'].astype(object)
dataset['pickup_year'] = dataset['pickup_year'].astype(int)

# Removing Outliers 

#Removing rows having passenger_counts <1, and >6
dataset["passenger_count"] = np.where(dataset["passenger_count"] > 6, np.nan, dataset['passenger_count'])
dataset["passenger_count"] = np.where(dataset["passenger_count"] < 1,np.nan, dataset['passenger_count'])

#Removing rows having fare_amount <0.1, and >500
dataset["fare_amount"] = np.where(dataset["fare_amount"] > 500, np.nan, dataset['fare_amount'])
dataset["fare_amount"] = np.where(dataset["fare_amount"] < 0.1 ,np.nan, dataset['fare_amount'])

#Removing rows having trip_distance <0.1, and >200
dataset["trip_distance"] = np.where(dataset["trip_distance"] > 200, np.nan, dataset['trip_distance'])
dataset["trip_distance"] = np.where(dataset["trip_distance"] <= 0 ,np.nan, dataset['trip_distance'])

missing_values = dataset.isnull().sum()
dataset = dataset.dropna()
dataset.info()

plt.bar(dataset["passenger_count"], dataset["fare_amount"], align='center', alpha=0.5)

# we do some EDA / check correlation between the variables of the data

trips_year_fareamount=dataset.groupby(['pickup_year'])['fare_amount'].mean().reset_index().rename(columns={'fare_amount':'avg_fare_amount'})
sns.barplot(x='pickup_year',y='avg_fare_amount',data=trips_year_fareamount).set_title("Avg Fare Amount over Years")

 trips_passenger_count_fareamount=dataset.groupby(['passenger_count'])['fare_amount'].mean().reset_index().rename(columns={'fare_amount':'avg_fare_amount'})
sns.barplot(x='passenger_count',y='avg_fare_amount',data=trips_year_fareamount).set_title("Avg Fare Amount over passenger count")

trips_hour_fareamount=dataset.groupby(['pickup_hour'])['fare_amount'].mean().reset_index().rename(columns={'fare_amount':'avg_fare_amount'})
sns.barplot(x='pickup_hour',y='avg_fare_amount',data=trips_hour_fareamount).set_title("Avg Fare Amount over Years")

trips_day_fareamount=dataset.groupby(['pickup_day'])['fare_amount'].mean().reset_index().rename(columns={'fare_amount':'avg_fare_amount'})
sns.barplot(x='pickup_day',y='avg_fare_amount',data=trips_day_fareamount).set_title("Avg Fare Amount over Years")

trips_month_fareamount=dataset.groupby(['pickup_month'])['fare_amount'].mean().reset_index().rename(columns={'fare_amount':'avg_fare_amount'})
sns.barplot(x='pickup_month',y='avg_fare_amount',data=trips_month_fareamount).set_title("Avg Fare Amount over Years")

#We remove the variables Month and day
dataset.drop('pickup_month', axis=1, inplace=True)
dataset.drop('pickup_day', axis=1, inplace=True)
dataset.drop('pickup_longitude', axis=1, inplace=True)

dataset_2 = dataset

X = dataset.iloc[:,1:].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# creating one hot encoder object with categorical feature 0 
# indicating the first column 

onehotencoder = OneHotEncoder(categorical_features = [1])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.fit_transform(X_test).toarray()


# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()

#model fitting - Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred_lm = regressor.predict(X_test)

#Performance Evaluation - #Mean absolute error - multiple linear
from sklearn import metrics
MAE_lm = metrics.mean_absolute_error(y_test,y_pred_lm)
#Mean squared error
MSE_lm = metrics.mean_squared_error(y_test,y_pred_lm)
#RSME
RSME_lm = np.sqrt(MSE)

#model fitting - Decision tree regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0, max_depth = 2, splitter = "best",
                                  min_samples_split = 25)
regressor.fit(X_train,y_train)
y_pred_dt = regressor.predict(X_test)

#Performance Evaluation 
from sklearn import metrics
MAE_dt = metrics.mean_absolute_error(y_test,y_pred_dt)
#Mean squared error
MSE_dt = metrics.mean_squared_error(y_test,y_pred_dt)
#RSME
RSME_dt = np.sqrt(MSE_dt)

#model fitting - Random Forest regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(max_depth = 5, n_estimators = 800,
                                  min_samples_split = 3)  
regressor.fit(X_train,y_train)
y_pred_rf = regressor.predict(X_test)

#Performance Evaluation 
from sklearn import metrics
MAE_rf = metrics.mean_absolute_error(y_test,y_pred_rf)
#Mean squared error
MSE_rf = metrics.mean_squared_error(y_test,y_pred_rf)
#RSME
RSME_rf = np.sqrt(MSE_rf)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#model fitting - SVM Regression
from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf', C = 10 , epsilon = 1, 
              shrinking = False, verbose = True)
svr_reg.fit(X_train,y_train)
y_pred_svm = svr_reg.predict(X_test)


#Performance Evaluation - #Mean absolute error - SVM
from sklearn import metrics
MAE_svm = metrics.mean_absolute_error(y_test,y_pred_svm)
#Mean squared error
MSE_svm = metrics.mean_squared_error(y_test,y_pred_svm)
#RSME
RSME_svm = np.sqrt(MSE_svm)


X = dataset_2.iloc[:,1:].values
y = dataset_2.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Gradient boosting Regression : 
params = {'n_estimators': 1000, 'max_depth': 2, 'min_samples_split': 25,
          'learning_rate': 0.01, 'loss': 'huber', 'min_samples_leaf': 3 }
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
y_pred_xgb = clf.predict(X_test)

#Performance Evaluation - #Mean absolute error - XGB
from sklearn import metrics
MAE_xgb = metrics.mean_absolute_error(y_test,y_pred_xgb)
#Mean squared error
MSE_xgb = metrics.mean_squared_error(y_test,y_pred_xgb)
#RSME
RSME_xgb = np.sqrt(MSE_xgb)




