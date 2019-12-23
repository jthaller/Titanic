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


# Cosmetics for data readability
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
