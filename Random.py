import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
import graphviz
import pydot
from sklearn.metrics import f1_score

# read data into dataframes
resources = pd.read_csv('C:/Users/prana/Downloads/DonorschooseDataset/resources.csv')
train = pd.read_csv('C:/Users/prana/Downloads/DonorschooseDataset/train.csv')

# combine dataframes
new = resources.groupby(['id']).sum().reset_index()
full = pd.merge(new, train)

# clean date
full['project_submitted_datetime'] = full.project_submitted_datetime.map(lambda p: p[:7])

# make predictors and target variables
full_features = ['quantity', 'price', 'project_subject_categories',
                 'school_state', 'project_grade_category', 'project_submitted_datetime'
                 ]
full_0 = full[full.project_is_approved == 0]
full_1 = full[full.project_is_approved == 1]
X_0 = full_0[full_features]
X_1 = full_1[full_features]
y_0 = full_0.project_is_approved
y_1 = full_1.project_is_approved
X = full[full_features]
y = full.project_is_approved

# split data
train_X_0, val_X_0, train_y_0, val_y_0 = train_test_split(X_0, y_0, random_state=0, train_size=.90)
train_X_1, val_X_1, train_y_1, val_y_1 = train_test_split(X_1, y_1, random_state=0, train_size=.35, test_size=.1)
train_X = pd.concat([train_X_0, train_X_1], ignore_index=True)
train_y = pd.concat([train_y_0, train_y_1], ignore_index=True)
val_X = pd.concat([val_X_0, val_X_1], ignore_index=True)
val_y = pd.concat([val_y_0, val_y_1], ignore_index=True)

# one hot encoding (cleaning up categorical values to make it easier to go through)
s = (train_X.dtypes == 'object')
object_cols = list(s[s].index)

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(val_X[object_cols]))

OH_cols_train.index = train_X.index
OH_cols_valid.index = val_X.index

num_X_train = train_X.drop(object_cols, axis=1)
num_X_valid = val_X.drop(object_cols, axis=1)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

model = RandomForestClassifier(random_state=1, n_estimators=400, max_depth=20, min_samples_leaf=20,
                               max_leaf_nodes=100000)
dt = model.fit(OH_X_train, train_y)
prediction = model.predict(OH_X_valid)
validation = model.predict(OH_X_train)

print(accuracy_score(val_y, prediction))
print(f1_score(val_y, prediction))
