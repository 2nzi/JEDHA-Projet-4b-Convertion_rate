#%%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier,VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score
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


#--------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------ Class -------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------

class ScoreLog:
    def __init__(self, save_score):
        self.save_score = save_score
        self.load_from_csv()

    def load_from_csv(self):
        if os.path.exists(self.save_score):
            self.df = pd.read_csv(self.save_score)
        else:
            self.df = pd.DataFrame(columns=["len_data", "model_name", "features_list", "f1_score_train", "f1_score_test", "hyperparameters", "datetime"])

    def log_score(self, len_data, model_name, features_list, f1_score_train, f1_score_test, hyperparameters):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = {"len_data": len_data, "model_name": model_name, "features_list": features_list, "f1_score_train": f1_score_train, "f1_score_test": f1_score_test, "hyperparameters": hyperparameters, "datetime": now}
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.save_to_csv()

    def save_to_csv(self):
        self.df.to_csv(self.save_score, index=False)

    def get_best_score(self, model_name=None):
        if model_name:
            filtered_df = self.df[self.df['model_name'] == model_name]
            if filtered_df.empty:
                return None  # Handle case where no scores for the specified model are found
            best_score_row = filtered_df.loc[filtered_df['f1_score_test'].idxmax()]
        else:
            best_score_row = self.df.loc[self.df['f1_score_test'].idxmax()]
        return best_score_row


class F1ScoreEvaluator:
    def __init__(self, classifier, classifier_name, X_train, X_test, Y_train, Y_test, param_grid={}, cv=3, scoring='f1', verbose=0):
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
        self.best_estimator_ = None
        self.f1_score_train = None
        self.f1_score_test = None
        self.cv_f1_scores = None 
    
    def find_best_params(self):
        # GridSearchCV to find the best hyperparameters
        gridsearch = GridSearchCV(self.classifier, param_grid=self.param_grid, cv=self.cv, scoring=self.scoring, verbose=self.verbose)
        gridsearch.fit(self.X_train, self.Y_train)
        
        # Storing the best parameters, best score, and best estimator
        self.best_params_ = gridsearch.best_params_
        self.best_score_ = gridsearch.best_score_
        self.best_estimator_ = gridsearch.best_estimator_
    
    def evaluate_train_test(self):
        # Calculating the F1 score on the training and test sets
        Y_train_pred = self.best_estimator_.predict(self.X_train)
        Y_test_pred = self.best_estimator_.predict(self.X_test)
        self.f1_score_train = f1_score(self.Y_train, Y_train_pred)
        self.f1_score_test = f1_score(self.Y_test, Y_test_pred)
        
        # Displaying the F1 scores
        print(f"{self.classifier_name} F1-score on train set: {self.f1_score_train}")
        print(f"{self.classifier_name} F1-score on test set: {self.f1_score_test}")
    
    def cross_validate(self):
        # Perform cross-validation and compute F1 scores
        cv_scores = cross_val_score(self.best_estimator_, self.X_train, self.Y_train, cv=self.cv, scoring=self.scoring)
        self.cv_f1_scores = cv_scores
        print(f"{self.classifier_name} Cross-validated F1 scores:", cv_scores)
        print(f"{self.classifier_name} Mean F1 score:", cv_scores.mean())

score_logger = ScoreLog('save_score_challenge.csv')







#%%
#--------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------ Data Process ------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------


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

# X = data.drop(target, axis=1)
X = data[features_list]
Y = data[target]
test_size_var=0.01
random_state_var = 42
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size_var, random_state=0, stratify=Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size_var, random_state=random_state_var, stratify=Y)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)



#%%
#--------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------ First Basic Model -------------------------------------------
#--------------------------------------------------------------------------------------------------------------------

classifiers = [
    (LogisticRegression(), 'LogisticRegression'),
    (RandomForestClassifier(), 'RandomForestClassifier'),
    (SVC(), 'SVC'),
    (AdaBoostClassifier(),'AdaBoostClassifier'),
    (XGBClassifier(),'XGBRegressorClassifier'),
    (GradientBoostingClassifier(),'GradientBoostingClassifier')
]


for classifier, classifier_name in classifiers:
    evaluator = F1ScoreEvaluator(classifier, classifier_name, X_train, X_test, Y_train, Y_test, param_grid={})
    evaluator.find_best_params() 
    evaluator.evaluate_train_test()  
    score_logger.log_score(len_data=len(data), model_name=evaluator.classifier_name, features_list=features_list, f1_score_train=evaluator.f1_score_train, f1_score_test=evaluator.f1_score_test, hyperparameters=evaluator.best_params_)


#%%
#--------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------ Logisctic Regression ----------------------------------------
#--------------------------------------------------------------------------------------------------------------------
reg_logistic_regression = LogisticRegression()

params_lr = {
    'penalty': ['l1'],
    'C': [0.1],
    'solver': ['saga'],
    'max_iter': [100],
}

# params_lr = {}

evaluator_lr = F1ScoreEvaluator(reg_logistic_regression, 'LogisticRegression', X_train, X_test, Y_train, Y_test, param_grid=params_lr, cv=5, verbose=2)
evaluator_lr.find_best_params()
evaluator_lr.evaluate_train_test()

score_logger.log_score(len_data=len(data), model_name=evaluator_lr.classifier_name, features_list=features_list, f1_score_train=evaluator_lr.f1_score_train, f1_score_test=evaluator_lr.f1_score_test, hyperparameters=evaluator_lr.best_params_)
# evaluator_lr.cross_validate()





#%%

#--------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------ Random Forest -----------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
reg_random_forest = RandomForestClassifier()

params_rf = {
    'max_depth': [10],
    'min_samples_leaf': [10],
    'min_samples_split': [4],
    'n_estimators': [100]
}

# params_rf = {
#     'max_depth': np.arange(5,12),
#     'min_samples_leaf': np.arange(2,6,2),
#     'min_samples_split': np.arange(2,6,2),
#     'n_estimators': [10 , 100]
# }

evaluator_rf = F1ScoreEvaluator(reg_random_forest, 'RandomForestClassifier', X_train, X_test, Y_train, Y_test, param_grid=params_rf, cv=5, verbose=2)
evaluator_rf.find_best_params()
evaluator_rf.evaluate_train_test()

score_logger.log_score(len_data=len(data), model_name=evaluator_rf.classifier_name, features_list=features_list, f1_score_train=evaluator_rf.f1_score_train, f1_score_test=evaluator_rf.f1_score_test, hyperparameters=evaluator_rf.best_params_)






#%%
#--------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------ SVC ---------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
reg_svc = SVC()

# Define parameter grid for SVC
# params_svc = {
#     'C': [0.1, 1, 10],
#     'kernel': ['rbf', 'poly'],
    # 'gamma': [0.1, 0.01, 0.001, 'scale', 'auto']
# }

# params_svc = {
#     'C': [0.1, 1],
#     'kernel': ['rbf'],
#     'gamma': ['scale', 'auto']
# }

params_svc = {
    'C': [1,3],
}

evaluator_svc = F1ScoreEvaluator(reg_svc, 'SVC', X_train, X_test, Y_train, Y_test, param_grid=params_svc, cv=5, verbose=2)
evaluator_svc.find_best_params()
evaluator_svc.evaluate_train_test()

score_logger.log_score(len_data=len(data), model_name=evaluator_svc.classifier_name, features_list=features_list, f1_score_train=evaluator_svc.f1_score_train, f1_score_test=evaluator_svc.f1_score_test, hyperparameters=evaluator_svc.best_params_)



#%%
#--------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------ XGBClassifier -----------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
xgboost = XGBClassifier()

# params = {
#     'max_depth': [4, 6, 10],
#     'min_child_weight': [ 2, 4, 6],
#     'n_estimators': [20]
# }

params = {
    'max_depth': [6],
    'min_child_weight': [4],
    'n_estimators': [15, 20,32]
}



evaluator_xgb = F1ScoreEvaluator(xgboost, 'XGBClassifier', X_train, X_test, Y_train, Y_test, param_grid=params, cv=5, verbose=2)
evaluator_xgb.find_best_params()
evaluator_xgb.evaluate_train_test()

score_logger.log_score(len_data=len(data), model_name=evaluator_xgb.classifier_name, features_list=features_list, f1_score_train=evaluator_xgb.f1_score_train, f1_score_test=evaluator_xgb.f1_score_test, hyperparameters=evaluator_xgb.best_params_)


#%%



#--------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------ AdaBoostClassifier -----------------------------------------
#--------------------------------------------------------------------------------------------------------------------

decision_tree = DecisionTreeClassifier()
adaboost_dt = AdaBoostClassifier(estimator=decision_tree)

params = {
    'estimator__max_depth': [4,6,8],
    'estimator__min_samples_leaf': [2, 4, 8],
    'estimator__min_samples_split': [12, 16],
    'n_estimators': [2, 4, 8]
}

evaluator_ada = F1ScoreEvaluator(adaboost_dt, 'AdaBoostClassifier', X_train, X_test, Y_train, Y_test, param_grid=params, cv=5, verbose=2)
evaluator_ada.find_best_params()
evaluator_ada.evaluate_train_test()

score_logger.log_score(len_data=len(data), model_name=evaluator_ada.classifier_name, features_list=features_list, f1_score_train=evaluator_ada.f1_score_train, f1_score_test=evaluator_ada.f1_score_test, hyperparameters=evaluator_ada.best_params_)


#%%
#--------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------ GradientBoostingClassifier ----------------------------------
#--------------------------------------------------------------------------------------------------------------------


gradientboost = GradientBoostingClassifier()

# params = {
#     'max_depth': [8, 10, 12],
#     'min_samples_leaf': [2, 4],
#     'min_samples_split': [6, 8, 10],
#     'n_estimators': [2, 4, 8, 12]
# }

params = {
    'max_depth': [8, 12],
    'min_samples_leaf': [10,14],
    'min_samples_split': [8],
    'n_estimators': [48, 64]
}

evaluator_gb = F1ScoreEvaluator(gradientboost, 'GradientBoostingClassifier', X_train, X_test, Y_train, Y_test, param_grid=params, cv=5, verbose=2)
evaluator_gb.find_best_params()
evaluator_gb.evaluate_train_test()

score_logger.log_score(len_data=len(data), model_name=evaluator_gb.classifier_name, features_list=features_list, f1_score_train=evaluator_gb.f1_score_train, f1_score_test=evaluator_gb.f1_score_test, hyperparameters=evaluator_gb.best_params_)


#%%
#--------------------------------------------------------------------------------------------------------------------#---------------------------------------- VotingClassifier -------------------------------------------
#-------------------------------------------------- VotingClassifier ----------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------


voting = VotingClassifier(
    # estimators=[("logistic", reg_logistic_regression), ("random_forest", reg_random_forest), ("svc", reg_svc)],
    # estimators=[("logistic", reg_logistic_regression), ("xgboost", xgboost), ("svc", reg_svc)],
    estimators=[("logistic", reg_logistic_regression), ("xgboost", xgboost), ("random_forest", reg_random_forest)],
    # estimators=[("logistic", reg_logistic_regression_best), ("xgboost", xgboost_best), ("random_forest", reg_random_forest_best)],
    voting="soft",
)

voting.fit(X_train, Y_train)

# y_pred_train = voting.predict(X_train)
# f1_train = f1_score(Y_test, y_pred_train)
# print("F1 score on train set:", f1_train)

y_pred_test = voting.predict(X_test)
f1_test = f1_score(Y_test, y_pred_test)

print("F1 score on test set:", f1_test)
print(voting)


# evaluator_voting = F1ScoreEvaluator(voting, 'voting', X_train, X_test, Y_train, Y_test, param_grid=params, cv=5, verbose=2)
# evaluator_voting.find_best_params()
# evaluator_voting.evaluate_train_test()
# score_logger.log_score(len_data=len(data), model_name='voting', features_list=features_list, f1_score_train=f1_train, f1_score_test=f1_test, hyperparameters={})





#%%
#--------------------------------------------------------------------------------------------------------------------#---------------------------------------- VotingClassifier -------------------------------------------
#-------------------------------------------------- StackingClassifier ----------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------

from sklearn.neural_network import MLPClassifier


reg_logistic_regression_best = LogisticRegression(penalty='l1', C=0.1, solver='saga', max_iter=100)
xgboost_best = XGBClassifier(n_estimators=20, max_depth=6, min_child_weight=4)
reg_random_forest_best = RandomForestClassifier(max_depth=10, min_samples_leaf=10, min_samples_split=4, n_estimators=100)
gradientboost_best = GradientBoostingClassifier(max_depth = 8,min_samples_leaf=10,min_samples_split = 8,n_estimators = 48)
adaboost_dt_best = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=6, min_samples_leaf=4, min_samples_split=12),n_estimators=2)
reg_svc_best = SVC()

stacking = StackingClassifier(
    # estimators=[("logistic", reg_logistic_regression_best), ("random_forest", reg_random_forest_best)],
    # estimators=[("logistic", reg_logistic_regression), ("xgboost", xgboost), ("random_forest", reg_random_forest)],
    # estimators=[("logistic", reg_logistic_regression_best), ("xgboost", xgboost_best), ("random_forest", reg_random_forest_best)],
    # estimators=[("logistic", reg_logistic_regression_best), ("gradientboost_best", gradientboost_best), ("random_forest", reg_random_forest_best)],
    # estimators=[("logistic", reg_logistic_regression_best), ("adaboost_dt_best", adaboost_dt_best), ("random_forest", reg_random_forest_best)],
    # estimators=[("logistic", reg_logistic_regression_best), ("reg_svc_best", reg_svc_best), ("random_forest", reg_random_forest_best)],
    estimators=[("logistic", reg_logistic_regression_best),  ("random_forest", reg_random_forest_best)],
    )

preds = stacking.fit_transform(X_train, Y_train)
predictions = pd.DataFrame(preds, columns=stacking.named_estimators_.keys())
# display(predictions)


stacking.fit(X_train, Y_train)

y_pred_train = stacking.predict(X_train)
y_pred_test = stacking.predict(X_test)

f1_train = f1_score(Y_train, y_pred_train)
f1_test = f1_score(Y_test, y_pred_test)

print("F1 score on training set:", f1_train)
print("F1 score on test set:", f1_test)

corr_matrix = predictions.corr().round(2)
import plotly.figure_factory as ff

fig = ff.create_annotated_heatmap(corr_matrix.values, x=corr_matrix.columns.tolist(), y=corr_matrix.index.tolist())
fig.show()






#%%
#--------------------------------------------------------------------------------------------------------------------#---------------------------------------- VotingClassifier -------------------------------------------
#-------------------------------------------------- Best Score (by model) ----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

name = {
    0: 'LogisticRegression',
    1: 'RandomForestClassifier',
    2: 'SVC',
    3: 'AdaBoostClassifier',
    4: 'XGBRegressorClassifier',
    5: 'GradientBoostingClassifier'
}

best_score_by_model = score_logger.get_best_score(model_name=name.get(3))
print("Best score for Logistic Regression model:", best_score_by_model)


#%%
#--------------------------------------------------------------------------------------------------------------------#---------------------------------------- VotingClassifier -------------------------------------------
#-------------------------------------------------- Best Score (over all model) ----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

best_score_row = score_logger.get_best_score()
print("Best Score:")
print(best_score_row)


















#%%


"""
Optuna example that optimizes a classifier configuration for Iris dataset using sklearn.

In this example, we optimize a classifier configuration for Iris dataset. Classifiers are from
scikit-learn. We optimize both the choice of classifier (among SVC and RandomForest) and their
hyperparameters.

"""

import optuna

import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=10
        )

    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(study.best_trial)



#%%

print("metrics training set")
print(display_metrics(Y_train, Y_train_pred))
print("metrics test set")
print(display_metrics(Y_test, Y_test_pred))