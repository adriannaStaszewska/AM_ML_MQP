#Get final results

#Regression model for porosity prediction
import numpy as np
import pandas as pd
from statistics import mean
import sklearn
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.model_selection import LeaveOneOut, LeavePOut, cross_validate, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, make_scorer
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

RANDOM_STATE = 42

features = ["laser_power", "scan_speed", "hatch_spacing"] #variables to predict on --must match the header names in the csv file
prediction = 'porosity' #feature to predict: 'porosity', 'porosity_lof' or 'porosity_keyhole' --must match the header in csv file

# load data
TEST = ['Q6', 'R6', 'G8', 'H6R', 'J3R', 'Q3']  #samples in the test set
df = pd.read_csv('/home/azstaszewska/Data/MS Data/sample_summary_v7.csv', ) #load dataset

df["test"] = np.where(df['set'].isin(TEST), True, False) #new column in the dataframe, is sample in the test set

#Set up train set
X_train = df[(df["test"]) == False]
y_train = X_train[prediction]*100
X_train = X_train[features]

#set up test set
X_test = df[df["test"] == True]
y_test = X_test[prediciton]*100
X_test = X_test[features]

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train) #apply standard scaling

#apply the same scaling to test set
X_test_scaled = scaler.transform(X_test)

#train the model
model = GradientBoostingRegressor(random_state = RANDOM_STATE, learning_rate = 0.01, loss="huber", n_estimators = 300).fit(X_train_scaled, y_train)

y_hat = model.predict(X_test_scaled) #make predicitons only on the test set
print(mean_absolute_percentage_error(y_test, y_hat)) #print out MAPE for the test set

#create an output file
results = pd.DataFrame()

results["sample"] = df["set"]
results['test'] = df["test"] #True if sample is in the test set, False otherwise
results["true_porosity"] = df[prediciton]*100
results["predicted_porosity"] = model.predict(df[features]) #make predicitons for the entire datset

#save to the output file
results.to_csv('/home/azstaszewska/Data/MS Data/final_results_1.csv')
