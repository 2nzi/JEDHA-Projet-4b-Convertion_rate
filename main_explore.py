#%%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    mean_absolute_error,
)
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC

warnings.filterwarnings(
    "ignore", category=DeprecationWarning
)  # to avoid deprecation warnings

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import os
import pandas as pd
from datetime import datetime


class ScoreLog:
    def __init__(self, save_score):
        self.save_score = save_score
        self.df = pd.DataFrame(columns=["len_data", "model_name", "features_list", "f1_score_train", "f1_score_test", "hyperparameters", "datetime"])

    def log_score(self, len_data, model_name, features_list, f1_score_train, f1_score_test, hyperparameters):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = {"len_data": len_data, "model_name": model_name, "features_list": features_list, "f1_score_train": f1_score_train, "f1_score_test": f1_score_test, "hyperparameters": hyperparameters, "datetime": now}
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.save_to_csv()

    def save_to_csv(self):
        mode = 'a' if os.path.exists(self.save_score) else 'w'
        self.df.to_csv(self.save_score, index=False, mode=mode, header=not os.path.exists(self.save_score))

    def get_best_score(self):
        return self.df.loc[self.df['f1_score_test'].idxmax()]


class F1ScoreEvaluator:
    def __init__(self, classifier, classifier_name, X_train, X_test, Y_train, Y_test, param_grid=None, cv=3, scoring='f1', verbose=0):
        self.classifier = classifier
        self.classifier_name = classifier_name
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.best_params_ = None
        self.best_score_ = None
        self.f1_score_train = None
        self.f1_score_test = None
    
    def find_best_params(self):
        # GridSearchCV pour trouver les meilleurs hyperparamètres
        gridsearch = GridSearchCV(self.classifier, param_grid=self.param_grid, cv=self.cv, scoring=self.scoring, verbose=self.verbose)
        gridsearch.fit(self.X_train, self.Y_train)
        
        # Stockage des meilleurs paramètres et score
        self.best_params_ = gridsearch.best_params_
        self.best_score_ = gridsearch.best_score_
        
        # Utilisation du meilleur classificateur trouvé
        self.classifier = gridsearch.best_estimator_
    
    def evaluate_train_test(self):
        # Calcul du score F1 sur les ensembles d'entraînement et de test
        Y_train_pred = self.classifier.predict(self.X_train)
        Y_test_pred = self.classifier.predict(self.X_test)
        self.f1_score_train = f1_score(self.Y_train, Y_train_pred)
        self.f1_score_test = f1_score(self.Y_test, Y_test_pred)
        
        # Affichage des scores F1
        print(f"{self.classifier_name} f1-score on train set : {self.f1_score_train}")
        print(f"{self.classifier_name} f1-score on test set : {self.f1_score_test}")


data = pd.read_csv('conversion_data_train.csv')
target = 'converted'

categorical_features = ['country', 'source', 'new_user']
numeric_features = ['age', 'total_pages_visited']
features_list = categorical_features+numeric_features

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)



######################################### DATA PROCESS ###############################################################
# X = data.drop(target, axis=1)
X = data[features_list]
Y = data[target]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)



######################################### LOGISTIC REGRESSION ###############################################################

# Define LogisticRegression

params_lr = {
    'penalty': ['l1'],
    'C': [0.1],
    'solver': ['saga'],
    'max_iter': [100],
    # 'tol': [1e-3, 1e-4, 1e-5], 
}

# params_lr = {}

reg_logistic_regression = LogisticRegression()

# Evaluate the LogisticRegression 
evaluator_lr = F1ScoreEvaluator(reg_logistic_regression, 'LogisticRegression', X_train, X_test, Y_train, Y_test, param_grid=params_lr, cv=5, verbose=2)
evaluator_lr.find_best_params()
evaluator_lr.evaluate_train_test()

# Log the scores
score_logger = ScoreLog('save_score_challenge.csv')
score_logger.log_score(len_data=len(data), model_name=evaluator_lr.classifier_name, features_list=features_list, f1_score_train=evaluator_lr.f1_score_train, f1_score_test=evaluator_lr.f1_score_test, hyperparameters=evaluator_lr.best_params_)




#%%

######################################### RANDOM FOREST ###############################################################
# Define RandomForestClassifier and its hyperparameter grid
params_rf = {
    'max_depth': [10],
    'min_samples_leaf': [10],
    'min_samples_split': [4],
    'n_estimators': [100, 200, 300, 400]
}
reg_random_forest = RandomForestClassifier()

# Evaluate the RandomForestClassifier with grid search
evaluator_rf = F1ScoreEvaluator(reg_random_forest, 'RandomForestClassifier', X_train, X_test, Y_train, Y_test, param_grid=params_rf, cv=5, verbose=2)
evaluator_rf.find_best_params()
evaluator_rf.evaluate_train_test()

# Log the scores
score_logger.log_score(len_data=len(data), model_name=evaluator_rf.classifier_name, features_list=features_list, f1_score_train=evaluator_rf.f1_score_train, f1_score_test=evaluator_rf.f1_score_test, hyperparameters=evaluator_rf.best_params_)

#%%
######################################### SVM ###############################################################

# Define parameter grid for SVC
# params_svc = {
#     'C': [0.1, 1, 10],
#     'kernel': ['rbf', 'poly'],
#     'gamma': ['scale', 'auto']
# }

params_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf'],
    'gamma': [0.1, 0.01, 0.001, 'scale', 'auto']
}

params_svc = {
    'C': [0.1, 1],
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto']
}
params_svc = {
    'C': [0.1, 1],
}
# Initialize SVC classifier
reg_svc = SVC()

# Evaluate SVC with grid search
evaluator_svc = F1ScoreEvaluator(reg_svc, 'SVC', X_train, X_test, Y_train, Y_test, param_grid=params_svc, cv=5, verbose=2)
evaluator_svc.find_best_params()
evaluator_svc.evaluate_train_test()

# Log the scores
score_logger.log_score(len_data=len(data), model_name=evaluator_svc.classifier_name, features_list=features_list, f1_score_train=evaluator_svc.f1_score_train, f1_score_test=evaluator_svc.f1_score_test, hyperparameters=evaluator_svc.best_params_)







#%%
######################################### BEST SCORE MODEL ###############################################################

best_score_row = score_logger.get_best_score()
print("Best Score:")
print(best_score_row)






















#%%
# classifiers = [
#     (LogisticRegression(max_iter=1000), 'LogisticRegression')
#     (RandomForestClassifier(), 'RandomForestClassifier'),
#     (SVC(), 'SVC')
# ]

# for classifier, classifier_name in classifiers:
#     evaluator = F1ScoreEvaluator(classifier, classifier_name, X_train, X_test, Y_train, Y_test)
#     f1_score_train, f1_score_test = evaluator.evaluate()
#     score_logger.log_score(len_data=len(data), model_name=classifier_name, features_list=features_list, f1_score_train=f1_score_train, f1_score_test=f1_score_test, hyperparameters={"": ""})




active = False

if f1_score_best < f1_score_test and active:
    f1_score_best = f1_score_test

    X = np.append(X_train,X_test,axis=0)
    Y = np.append(Y_train,Y_test)

    classifier.fit(X,Y)
    data_without_labels = pd.read_csv('conversion_data_test.csv')
    X_without_labels = data_without_labels.loc[:, features_list]

    # X_without_labels = X_without_labels.values
    X_without_labels = preprocessor.transform(X_without_labels)

    data_pred = {
        'converted': classifier.predict(X_without_labels)
    }

    Y_predictions = pd.DataFrame(columns=['converted'],data=data_pred)
    Y_predictions.to_csv('conversion_data_test_predictions_AntoineV-model3.csv', index=False)
    print('new best classifier',classifier_name,f1_score_best)


# %%
