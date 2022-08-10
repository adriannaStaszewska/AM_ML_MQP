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

# load data
TEST = ['Q6', 'R6', 'G8', 'H6R', 'J3R', 'Q3']
df = pd.read_csv('/home/azstaszewska/Data/MS Data/sample_summary_v7.csv', )
df["test"] = np.where(df['set'].isin(TEST), True, False)
X_train = df[(df["test"]) == False]
y_train = X_train['porosity']*100
X_train = X_train[["laser_power", "scan_speed", "hatch_spacing"]]


X_test = df[df["test"] == True]
y_test = X_test['porosity']*100
X_test = X_test[["laser_power", "scan_speed", "hatch_spacing"]]

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

model = GradientBoostingRegressor(random_state = RANDOM_STATE, learning_rate = 0.01, loss="huber", n_estimators = 300).fit(X_train, y_train)

X_test_scaled = scaler.transform(X_test)

y_hat = model.predict(X_test)
print(mean_absolute_percentage_error(y_test, y_hat))


results = pd.DataFrame()

results["sample"] = df["set"]
results["true_porosity"] = df['porosity']*100
results["predicted_porosity"] = model.predict(df[["laser_power", "scan_speed", "hatch_spacing"]])

results.to_csv('/home/azstaszewska/Data/MS Data/final_results_1.csv')
