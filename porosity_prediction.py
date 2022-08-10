#Regression model for porosity prediction
import numpy as np
import pandas as pd
from statistics import mean, stdev
import sklearn
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.model_selection import LeaveOneOut, LeavePOut, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, make_scorer
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
id = "all_v5_new_17"
RESULTS_FOLDER = "/home/azstaszewska/Data/MS Data/Results/"

RANDOM_STATE = 42
#define custom function which returns single output as metric score
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#make scorer from custome function
mape_scorer = make_scorer(MAPE)
print(sorted(sklearn.metrics.SCORERS.keys()))

# load data
df = pd.read_csv('/home/azstaszewska/Data/MS Data/sample_summary_v6.csv', )
df = df.dropna()
X = df[["laser_power", "scan_speed", "hatch_spacing"]]

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

y = 100-df["porosity_lof"]*100
#set up model
cv = LeavePOut(p=2)


scoring = ["neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_absolute_error", "neg_mean_absolute_percentage_error", "r2"]
scoring = {"neg_mean_squared_error": "neg_mean_squared_error",
"neg_root_mean_squared_error": "neg_root_mean_squared_error",
"neg_mean_absolute_error":"neg_mean_absolute_error",
"r2":"r2",
"neg_mean_absolute_percentage_error": make_scorer(MAPE)
}


print("--- Linear Regression ---")
model = LinearRegression()
scores = cross_validate(model, X_scaled, y, scoring=scoring,
                         cv=cv, n_jobs=-1)
print("MSE: " + str(mean(abs(scores["test_neg_mean_squared_error"]))))
print("MAE: " + str(mean(abs(scores["test_neg_mean_absolute_error"]))))
print("RMSE: " + str(mean(abs(scores["test_neg_root_mean_squared_error"]))))
print("MAPE: " + str(mean(abs(scores["test_neg_mean_absolute_percentage_error"]))))


print("MSE std: " + str(stdev(abs(scores["test_neg_mean_squared_error"]))))
print("MAE std: " + str(stdev(abs(scores["test_neg_mean_absolute_error"]))))
print("RMSE std: " + str(stdev(abs(scores["test_neg_root_mean_squared_error"]))))
print("MAPE std: " + str(stdev(abs(scores["test_neg_mean_absolute_percentage_error"]))))




print()
print()

print("--- Elastic Net ---")
'''
model = ElasticNet(random_state = RANDOM_STATE)
params = [{'alpha': [0.8, 0.85, 0.9, 1], 'l1_ratio': [ 0.5, 0.55, 0.6, 0.65, 0.7]}]
grid = GridSearchCV(model, params, refit=False, scoring = scoring, return_train_score=True, cv=cv)
grid.fit(X_scaled, y)
#print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
df_results = pd.DataFrame(grid.cv_results_)
df_results.to_csv(RESULTS_FOLDER+"elastic_cv_results_all_"+id+".csv")

df_trimmed = df_results[df_results.columns.drop(list(df_results.filter(regex='split')))]
df_trimmed.to_csv(RESULTS_FOLDER+"elastic_cv_results_trim_"+id+".csv")
'''
model1 = ElasticNet(random_state = RANDOM_STATE, alpha=0.8, l1_ratio = 0.3)
scores = cross_validate(model1, X_scaled, y, scoring=scoring,
                         cv=cv, n_jobs=-1)
print("MSE: " + str(mean(abs(scores["test_neg_mean_squared_error"]))))
print("MAE: " + str(mean(abs(scores["test_neg_mean_absolute_error"]))))
print("RMSE: " + str(mean(abs(scores["test_neg_root_mean_squared_error"]))))
print("MAPE: " + str(mean(abs(scores["test_neg_mean_absolute_percentage_error"]))))

print("MSE std: " + str(stdev(abs(scores["test_neg_mean_squared_error"]))))
print("MAE std: " + str(stdev(abs(scores["test_neg_mean_absolute_error"]))))
print("RMSE std: " + str(stdev(abs(scores["test_neg_root_mean_squared_error"]))))
print("MAPE std: " + str(stdev(abs(scores["test_neg_mean_absolute_percentage_error"]))))


'''
scores = cross_validate(model, X_scaled, y, scoring=scoring,
                         cv=cv, n_jobs=-1)

print("MSE: " + str(mean(abs(scores["test_neg_mean_squared_error"]))))
print("MAE: " + str(mean(abs(scores["test_neg_mean_absolute_error"]))))
print("RMSE: " + str(mean(scores["test_neg_root_mean_squared_error"])))

scores = cross_validate(model, X_scaled, y, scoring=mape_scorer,
                         cv=cv, n_jobs=-1)
print("MAPE: " + str(mean(scores["test_score"])))
'''

print()
print()

print("--- SVR ---")
model = SVR()
'''
params =[{'C': [0.1, 1, 1000], 'gamma': [0.1, 0.0001], 'kernel': ['rbf', 'poly', 'sigmoid']}]

grid = GridSearchCV(model, params, refit=False, scoring = scoring, return_train_score=True, cv=cv)
grid.fit(X_scaled, y)

df_results = pd.DataFrame(grid.cv_results_)
df_results.to_csv(RESULTS_FOLDER+"svr_cv_results_all_"+id+".csv")

df_trimmed = df_results[df_results.columns.drop(list(df_results.filter(regex='split')))]
df_trimmed.to_csv(RESULTS_FOLDER+"svr_cv_results_trim_"+id+".csv")

'''
model1 = SVR(C = 1000, gamma=0.000000001, kernel="rbf")
scores = cross_validate(model1, X_scaled, y, scoring=scoring,
                         cv=cv, n_jobs=-1)
print("MSE: " + str(mean(abs(scores["test_neg_mean_squared_error"]))))
print("MAE: " + str(mean(abs(scores["test_neg_mean_absolute_error"]))))
print("RMSE: " + str(mean(abs(scores["test_neg_root_mean_squared_error"]))))
print("MAPE: " + str(mean(abs(scores["test_neg_mean_absolute_percentage_error"]))))

print("MSE std: " + str(stdev(abs(scores["test_neg_mean_squared_error"]))))
print("MAE std: " + str(stdev(abs(scores["test_neg_mean_absolute_error"]))))
print("RMSE std: " + str(stdev(abs(scores["test_neg_root_mean_squared_error"]))))
print("MAPE std: " + str(stdev(abs(scores["test_neg_mean_absolute_percentage_error"]))))

'''
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

scores = cross_validate(grid_result, X_scaled, y, scoring=scoring,
                         cv=cv, n_jobs=-1)

print("MSE: " + str(mean(abs(scores["test_neg_mean_squared_error"]))))
print("MAE: " + str(mean(abs(scores["test_neg_mean_absolute_error"]))))
print("RMSE: " + str(mean(scores["test_neg_root_mean_squared_error"])))

scores = cross_validate(grid_result, X_scaled, y, scoring=mape_scorer,
                         cv=cv, n_jobs=-1)
print("MAPE: " + str(mean(scores["test_score"])))

'''
print()
print()
print("--- Bayesian Ridge Regressor ---")

model = BayesianRidge()
params = [{'alpha_1': [ 0.00001, 0.0000001, 0.000001, 0.00000001],  'alpha_2': [ 0.00001, 0.0000001, 0.000001, 0.00000001], 'lambda_1': [ 0.00001, 0.0000001, 0.000001, 0.00000001], 'lambda_2': [ 0.0000001, 0.000001, 0.00000001]}]

grid= GridSearchCV(model, params, refit=False, scoring = scoring, return_train_score=True, cv=cv)
grid.fit(X_scaled, y)

df_results = pd.DataFrame(grid.cv_results_)
df_results.to_csv(RESULTS_FOLDER+"bayesian_cv_results_all_"+id+".csv")

df_trimmed = df_results[df_results.columns.drop(list(df_results.filter(regex='split')))]
df_trimmed.to_csv(RESULTS_FOLDER+"bayesian_cv_results_trim_"+id+".csv")

model = BayesianRidge()
scores = cross_validate(model, X_scaled, y, scoring=scoring,
                         cv=cv, n_jobs=-1)

print("MSE: " + str(mean(abs(scores["test_neg_mean_squared_error"]))))
print("MAE: " + str(mean(abs(scores["test_neg_mean_absolute_error"]))))
print("RMSE: " + str(mean(abs(scores["test_neg_root_mean_squared_error"]))))
print("MAPE: " + str(mean(abs(scores["test_neg_mean_absolute_percentage_error"]))))

print("MSE std: " + str(stdev(abs(scores["test_neg_mean_squared_error"]))))
print("MAE std: " + str(stdev(abs(scores["test_neg_mean_absolute_error"]))))
print("RMSE std: " + str(stdev(abs(scores["test_neg_root_mean_squared_error"]))))
print("MAPE std: " + str(stdev(abs(scores["test_neg_mean_absolute_percentage_error"]))))

print()
print()

print("--- SGD Regressor ---")

model = SGDRegressor(random_state = RANDOM_STATE)
params = [{'loss': ['squared_error', 'epsilon_insensitive'], 'penalty': ['l1', 'l2'], 'alpha': [ 0.001, 0.01, 0.1] , 'learning_rate': ['constant', 'optimal', 'adaptive'], 'eta0':[1, 0.1, 0.01, 0.001, 0.0001] }]

grid= GridSearchCV(model, params, refit=False, scoring = scoring, return_train_score=True, cv=cv)
grid.fit(X_scaled, y)

df_results = pd.DataFrame(grid.cv_results_)
df_results.to_csv(RESULTS_FOLDER+"sgd_cv_results_all_"+id+".csv")

df_trimmed = df_results[df_results.columns.drop(list(df_results.filter(regex='split')))]
df_trimmed.to_csv(RESULTS_FOLDER+"sgd_cv_results_trim_"+id+".csv")


model1 =SGDRegressor(random_state = RANDOM_STATE, alpha=0.0000001, penalty='l1')
scores = cross_validate(model1, X_scaled, y, scoring=scoring,
                         cv=cv, n_jobs=-1)
print("MSE: " + str(mean(abs(scores["test_neg_mean_squared_error"]))))
print("MAE: " + str(mean(abs(scores["test_neg_mean_absolute_error"]))))
print("RMSE: " + str(mean(abs(scores["test_neg_root_mean_squared_error"]))))
print("MAPE: " + str(mean(abs(scores["test_neg_mean_absolute_percentage_error"]))))

print("MSE std: " + str(stdev(abs(scores["test_neg_mean_squared_error"]))))
print("MAE std: " + str(stdev(abs(scores["test_neg_mean_absolute_error"]))))
print("RMSE std: " + str(stdev(abs(scores["test_neg_root_mean_squared_error"]))))
print("MAPE std: " + str(stdev(abs(scores["test_neg_mean_absolute_percentage_error"]))))


#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

'''
scores = cross_validate(grid_result, X_scaled, y, scoring=scoring,
                         cv=cv, n_jobs=-1)

print("MSE: " + str(mean(abs(scores["test_neg_mean_squared_error"]))))
print("MAE: " + str(mean(abs(scores["test_neg_mean_absolute_error"]))))
print("RMSE: " + str(mean(scores["test_neg_root_mean_squared_error"])))

scores = cross_validate(grid_result, X_scaled, y, scoring=mape_scorer,
                         cv=cv, n_jobs=-1)
print("MAPE: " + str(mean(scores["test_score"])))

'''


print()
print()


print("--- Gradient Boosting Regressor ---")

model = GradientBoostingRegressor(random_state = RANDOM_STATE)
params = [{'loss': ['squared_error', 'absolute_error', 'huber'],  'learning_rate': [ 0.01, 0.05, 0.1, 0.5, 1], 'n_estimators': [100, 500]}]

grid= GridSearchCV(model, params, refit=False, scoring = scoring, return_train_score=True, cv=cv)
grid.fit(X_scaled, y)

df_results = pd.DataFrame(grid.cv_results_)
df_results.to_csv(RESULTS_FOLDER+"gradboost_cv_results_all_"+id+".csv")

df_trimmed = df_results[df_results.columns.drop(list(df_results.filter(regex='split')))]
df_trimmed.to_csv(RESULTS_FOLDER+"gradboost_cv_results_trim_"+id+".csv")


model1 = GradientBoostingRegressor(random_state = RANDOM_STATE, n_estimators = 200, learning_rate = 0.01)
scores = cross_validate(model1, X_scaled, y, scoring=scoring,
                         cv=cv, n_jobs=-1)
print("MSE: " + str(mean(abs(scores["test_neg_mean_squared_error"]))))
print("MAE: " + str(mean(abs(scores["test_neg_mean_absolute_error"]))))
print("RMSE: " + str(mean(abs(scores["test_neg_root_mean_squared_error"]))))
print("MAPE: " + str(mean(abs(scores["test_neg_mean_absolute_percentage_error"]))))

print("MSE std: " + str(stdev(abs(scores["test_neg_mean_squared_error"]))))
print("MAE std: " + str(stdev(abs(scores["test_neg_mean_absolute_error"]))))
print("RMSE std: " + str(stdev(abs(scores["test_neg_root_mean_squared_error"]))))
print("MAPE std: " + str(stdev(abs(scores["test_neg_mean_absolute_percentage_error"]))))

'''
model2 = GradientBoostingRegressor(random_state = RANDOM_STATE,  learning_rate = 0.0001)

scores = cross_validate(model2, X_scaled, y, scoring=scoring,
                         cv=cv, n_jobs=-1)

print("MSE: " + str(mean(abs(scores["test_neg_mean_squared_error"]))))
print("MAE: " + str(mean(abs(scores["test_neg_mean_absolute_error"]))))
print("RMSE: " + str(mean(scores["test_neg_root_mean_squared_error"])))

scores = cross_validate(model2, X_scaled, y, scoring=mape_scorer,
                         cv=cv, n_jobs=-1)
print("MAPE: " + str(mean(scores["test_score"])))
'''
'''
print()
print()
'''

'''
params = [{'loss': ['squared_loss', 'huber', 'absolute_loss', 'quantile'],  'learning_rate': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'n_estimators': [100, 500, 1000, 5000, 10000],'criterion': ["friedman_mse", "squared_loss","mse"] }]

grid= GridSearchCV(model, params, refit=False, scoring = scoring, return_train_score=True, cv=cv)
grid.fit(X_scaled, y)

df_results = pd.DataFrame(grid.cv_results_)
df_results.to_csv(RESULTS_FOLDER+"bayesian_cv_results_all_"+id+".csv")

df_trimmed = df_results[df_results.columns.drop(list(df.filter(regex='split')))]
df_trimmed.to_csv(RESULTS_FOLDER+"bayesian_cv_results_trim_"+id+".csv")
'''
