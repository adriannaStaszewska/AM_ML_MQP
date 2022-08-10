#Processing the data and running various classifcaiton models on the data
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score
from sklearn.naive_bayes import GaussianNB
import pickle
import sys
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

RESULTS_FOLDER = "/home/azstaszewska/Data/MS Data/Results/"
model_type = sys.argv[1]

df = pd.read_csv('/home/azstaszewska/Data/MS Data/extracted_features_clean_v4.csv')
df = df.dropna()
df = shuffle(df, random_state=RANDOM_STATE)
#train, test = train_test_split(df, test_size=0.1, random_state=RANDOM_STATE)

train_keyhole = df[df['class'] == 1]
train_lof = df[df['class'] == 0]
train_lof = resample(train_lof, n_samples=len(train_keyhole), random_state=RANDOM_STATE)
train = pd.concat([train_lof, train_keyhole])
train = shuffle(train, random_state=RANDOM_STATE)

cv = KFold(n_splits=10)
headers = ["sample", "polygon", "class", "area", "perimeter", "major_axis", "minor_axis", "convex_hull","ch_area", "centroid_x", "centriod_y", "max_feret_diameter", "min_feret_diameter", "bbox_x", "bbox_y", "bbox_h", "bbox_w", "bbox_ratio", "circularity", "aspect_ratio", "roundness", "solidity", "roughness", "shape_factor", "convexity", "perimeter_to_area_ratio", "ellipse_major_axis", "ellipse_minor_axis", "ellipse_angle", "ellipse_x", "ellipse_y", "ellipse_ratio", "laser_power", "scan_speed", "layer_thickness", "hatch_spacing", "sim_melt_pool_width", "linear_energy_density", "surface_energy_density", "volumetric_energy_density"]

features = ["area", "perimeter", "major_axis", "minor_axis", "ch_area", "centroid_x", "centriod_y", "max_feret_diameter", "max_feret_diameter", "bbox_x", "bbox_y", "bbox_h", "bbox_w", "bbox_ratio", "circularity", "aspect_ratio", "roundness", "solidity", "roughness", "shape_factor", "convexity", "perimeter_to_area_ratio", "ellipse_major_axis", "ellipse_minor_axis", "ellipse_angle", "ellipse_x", "ellipse_y", "ellipse_ratio"]
processing_parmas = ["laser_power", "scan_speed", "hatch_spacing", "sim_melt_pool_width", "linear_energy_density", "surface_energy_density", "volumetric_energy_density"]
not_correlated = ["laser_power", "scan_speed", "sim_melt_pool_width" ]
#selected = ["roughness", "shape_factor", "perimeter_to_area_ratio"
#selected = [ "area", "perimeter", "major_axis", "minor_axis", "bbox_ratio", "circularity", "aspect_ratio", "solidity", "roughness", "shape_factor",  "perimeter_to_area_ratio", "ellipse_major_axis", "ellipse_minor_axis", "ellipse_ratio"]
selected = ["bbox_ratio", "shape_factor", "perimeter_to_area_ratio", "aspect_ratio", "scan_speed"]
y_train = train.loc[:,"class"]
X_train = train.loc[:, selected]
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)



'''
bestK = SelectKBest(score_func=mutual_info_classif, k=6)
fit = bestK.fit(X_train, y_train)
cols = bestK.get_support(indices=True)
X_train = X_train.iloc[:,cols]
print(list(X_train.columns))
'''

metrics = ["accuracy", "average_precision", "f1", "r2"]
id = "custom_top4_v5_11"

#### KNN ####
if model_type == "knn" or model_type =="all":
    params =[{"n_neighbors": list(range(1, 2*20, 2)), "metric": ["euclidean", "manhattan", "minkowski"]}]
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, params, refit=False, scoring =metrics, return_train_score=True, cv=cv)
    clf.fit(X_train, y_train)
    df_results = pd.DataFrame(clf.cv_results_)
    df_results.to_csv(RESULTS_FOLDER+"knn_cv_results_"+id+".csv")


#### SGD ####
if model_type == "sgd" or model_type =="all":
    params = [{'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge','perceptron'], 'penalty': ['l1', 'l2', 'elasticnet'], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'], 'eta0':[0.1, 0.01, 0.001, 0.0001, 0.00001] }]
    sgd = SGDClassifier(max_iter = 100000)
    clf = GridSearchCV(sgd, params, refit=False, scoring = metrics, return_train_score=True, cv=cv)
    clf.fit(X_train, y_train)
    df_results = pd.DataFrame(clf.cv_results_)
    df_results.to_csv(RESULTS_FOLDER+"sgd_cv_results_"+id+".csv")

#### Decision Tree ####
if model_type == "tree" or model_type =="all":
    params = [{'criterion': ['gini', 'entropy'], 'max_depth':  list(range(2, 20))}]
    decision_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    clf = GridSearchCV(decision_tree, params, refit=False, scoring =metrics, return_train_score=True, cv=cv)
    clf.fit(X_train, y_train)
    df_results = pd.DataFrame(clf.cv_results_)
    df_results.to_csv(RESULTS_FOLDER+"tree_cv_results_"+id+".csv")

#### Random Forest ####
if model_type == "forest" or model_type =="all":
    params = {'n_estimators': [10, 25, 50, 75, 100, 125], 'max_features': [2, 3, 4],
     'max_depth': [10, 25, 50, None], 'bootstrap': [True, False]}
    random_forest = RandomForestClassifier(random_state=RANDOM_STATE)
    clf = GridSearchCV(random_forest, params, refit=False, scoring = metrics, return_train_score=True, cv=cv)
    clf.fit(X_train, y_train)
    df_results = pd.DataFrame(clf.cv_results_)
    df_results.to_csv(RESULTS_FOLDER+"forest_cv_results_"+id+".csv")



#### SVM ####
if model_type == "svm" or model_type =="all":
    params = [{"C": [ 1, 10, 100], 'kernel': ['linear']},
    {'C':  [1, 10, 100], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']},]
    svm = svm.SVC(cache_size=2000)
    clf = GridSearchCV(svm, params, refit=False, scoring =metrics, return_train_score=True, cv=cv)
    clf.fit(X_train, y_train)
    print(clf.n_splits_)
    df_results = pd.DataFrame(clf.cv_results_)
    df_results.to_csv(RESULTS_FOLDER+"svm_cv_results_"+id+".csv")
#save clf.cv_results_ to file

'''
#### Naive Bayes ####
clf = GaussianNB()
clf = GridSearchCV(random_forest, [], refit=True, scoring = "accuracy")
clf.fit(X_train, y_train)
df_results = pd.DataFrame(clf.cv_results_)
df_results.to_csv(RESULTS_FOLDER+"knn_cv_results.csv")
'''
