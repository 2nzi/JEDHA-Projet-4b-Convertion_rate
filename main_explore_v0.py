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



########################################################################################################""

data = pd.read_csv('conversion_data_train.csv')
target = 'converted'


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

# Example Usage
# score_logger.log_score(len_data=1000, model_name='Logistic Regression', features_list='age, gender, income', f1_score_train=0.85, f1_score_test=0.82, hyperparameters={'C': 0.1, 'penalty': 'l2'})
# score_logger.log_score(len_data=1200, model_name='Random Forest', features_list='age, gender, income', f1_score_train=0.88, f1_score_test=0.84, hyperparameters={'n_estimators': 100})

# best_score_row = score_logger.get_best_score()
# print("Best Score:")
# print(best_score_row)


class F1ScoreEvaluator:
    def __init__(self, classifier, classifier_name, X_train, X_test, Y_train, Y_test):
        self.classifier = classifier
        self.classifier_name = classifier_name
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
    
    def evaluate(self):
        self.classifier.fit(self.X_train, self.Y_train)
        Y_train_pred = self.classifier.predict(self.X_train)
        Y_test_pred = self.classifier.predict(self.X_test)
        f1_score_train = f1_score(self.Y_train, Y_train_pred)
        f1_score_test = f1_score(self.Y_test, Y_test_pred)
        print(f"{self.classifier_name} f1-score on train set : {f1_score_train}")
        print(f"{self.classifier_name} f1-score on test set : {f1_score_test}")
        return f1_score_train, f1_score_test



f1_score_best = 0 

categorical_features = ['country', 'source', 'new_user']
numeric_features = ['age', 'total_pages_visited']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Preprocess your data
X = data.drop(target, axis=1)
Y = data[target]
features_list = X.columns

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)






# Define classifiers
# classifier, classifier_name = LogisticRegression(max_iter=1000), 'LogisticRegression'
# classifier, classifier_name = RandomForestClassifier(), 'RandomForestClassifier'
classifier, classifier_name = RandomForestClassifier(), 'RandomForestClassifier'







# Evaluate classifiers
score_logger = ScoreLog('save_score_challenge.csv')
evaluator = F1ScoreEvaluator(classifier, classifier_name, X_train, X_test, Y_train, Y_test)
f1_score_train, f1_score_test = evaluator.evaluate()
score_logger.log_score(len_data=len(data), model_name=classifier_name, features_list=features_list, f1_score_train=f1_score_train, f1_score_test=f1_score_test, hyperparameters={"": ""})


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
