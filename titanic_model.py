# import classes and set cosmetics. Just making this one class to clean up the main file....
# Machine learning and data analysis project for the Titanic porject on Kaggle
# This script is based off the tutorial from:
# https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python

# after this try 1 feature engineering, models, hyper parameter optimation, stacking
#  higher scores gets a better grade.


# Import classes
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
import os


## Cosmetics for data readability
sns.set(style='white', context='notebook', palette='deep')
pylab.rcParams['figure.figsize'] = 18,12
warnings.filterwarnings('ignore')
mpl.style.use('ggplot')
sns.set_style('white')
%matplotlib inline

# load the data
test_filepath = "C:/Users/jerem/OneDrive/Documents/Python/titanic/test.csv"
train_filepath = "C:/Users/jerem/OneDrive/Documents/Python/titanic/train.csv"
df_train = pd.read_csv(train_filepath)
df_test = pd.read_csv(test_filepath)
df_train.head()


# Modify the graph above by assigning each species an individual color.
g = sns.FacetGrid(df_train, hue="Survived", col="Pclass", margin_titles=True,
                  palette={1:"seagreen", 0:"gray"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();
# See the relationship between the Age and fair. i.e. helps see things like if older people pay more for their ticket
df_train.plot(kind='scatter', x='Age', y='Fare', alpha = 0.5,color = 'red')
#show scatter plot with using Matplotlib
plt.figure(figsize=(8,6))
plt.scatter(range(df_train.shape[0]), np.sort(df_train['Age'].values))
plt.xlabel('index')
plt.ylabel('Age')
plt.title('Explore: Age')
plt.show()

# There are TONS of plots in this section. I'm going to skip this and come back to it. Some are definitely
# very useful, but doing them all takes up a lot of space. I'm going to skip that and maybe come back here.

# DATA preprocessing -----------------------------------
print(df_train.shape)


def check_missing_data(df):
    flag=df.isna().sum().any() # .any makes it return a boolean
    if flag==True:
        total = df.isnull().sum()
        percent = (df.isnull().sum())/(df.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        # written by MJ Bahmani
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)

df_train.shape
df_train.info()
df_train['Age'].unique()
df_train["Pclass"].value_counts()
df_train[df_train['Age']==30]

## Data Cleaning ------------------------------

#split numerical ages into categories
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

#split fare into quartiles
def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df

def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

df_train = transform_features(df_train)
df_test = transform_features(df_test)

df_train.head()

## Feature Encoding --------------------------------------------
#using LabelEncoder
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test


#Encode Dataset
df_train, df_test = encode_features(df_train, df_test)
df_train.head()

## Preventing overfitting
x_all = df_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = df_train['Survived']

num_test = 0.3
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test, random_state=100)

## RandomForestClassifier-----------------------
# Choose the type of classifier.
rfc = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9],
              'max_features': ['log2', 'sqrt','auto'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# notes from me about grid search:
# Grid search is the process of performing hyper parameter tuning in order to determine
#the optimal values for a given model. This is significant as the performance of the entire
# model is based on the hyper parameter values specified.
# Run the grid search
grid_obj = GridSearchCV(rfc, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
rfc = grid_obj.best_estimator_

# Fit the best algorithm to the data.
rfc.fit(X_train, y_train)

## Prediction --------------------------
rfc_prediction = rfc.predict(X_test)
rfc_score=accuracy_score(y_test, rfc_prediction)
print(rfc_score)


## XGBoost ----------------------
#booster [default=gbtree] change to gblinear to see. gbtree almost always outperforms though
xgboost = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
## Predictions ----------------
xgb_prediction = xgboost.predict(X_test)
xgb_score=accuracy_score(y_test, xgb_prediction)
print(xgb_score)


## Logistic Regression -------------------------
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
#Predictions ----------------------
logreg_prediction = logreg.predict(X_test)
logreg_score=accuracy_score(y_test, logreg_prediction)
print(logreg_score)


## Decision Tree Regressor ----------------------
from sklearn.tree import DecisionTreeRegressor
# Define model. Specify a number for random_state to ensure same results each run
dt = DecisionTreeRegressor(random_state=1)
# Fit model
dt.fit(X_train, y_train)
#preictions -----------------------
dt_prediction = dt.predict(X_test)
dt_score=accuracy_score(y_test, dt_prediction)
print(dt_score)


## Extra Tree Regressor
from sklearn.tree import ExtraTreeRegressor
# Define model. Specify a number for random_state to ensure same results each run
etr = ExtraTreeRegressor()
# Fit model
etr.fit(X_train, y_train)
etr_prediction = etr.predict(X_test)
etr_score=accuracy_score(y_test, etr_prediction)
print(etr_score)

## End of example project -----------------------------------------------------------------------------------------

## Hyperparameter optimization --------
# changing eta is equivilent to learning rate for gbm
#increasing max dpeth improves model to 6. 6 is actually the default but it does better
xgboost = xgb.XGBClassifier(max_depth=6, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
## Predictions ----------------
xgb_prediction = xgboost.predict(X_test)
xgb_score=accuracy_score(y_test, xgb_prediction)
print(xgb_score)
#messing with subsample doesn't help. default of 1 is best

#reduce the fraction of observations to be randomly samples for each tree from default of 1 to .25
xgboost = xgb.XGBClassifier(max_depth=6, n_estimators=300, learning_rate=0.05,colsample_bytree=.25).fit(X_train, y_train)
## Predictions ----------------
xgb_prediction = xgboost.predict(X_test)
xgb_score=accuracy_score(y_test, xgb_prediction)
print(xgb_score)



# 7-10 How Do I Submit?
# Fork and Commit this Kernel.
# Then navigate to the Output tab of the Kernel and "Submit to Competition".
# X_train = df_train.drop("Survived",axis=1)
# y_train = df_train["Survived"]
# X_train = X_train.drop("PassengerId",axis=1)
# X_test  = df_test.drop("PassengerId",axis=1)
# xgboost = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
# Y_pred = xgboost.predict(X_test)
# You can change your model and submit the results of other models
#
# submission = pd.DataFrame({
#         "PassengerId": df_test["PassengerId"],
#         "Survived": Y_pred
#     })
# submission.to_csv('submission.csv', index=False)
