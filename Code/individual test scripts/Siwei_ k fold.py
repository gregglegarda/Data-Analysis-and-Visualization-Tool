
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier #https://www.datacamp.com/community/tutorials/decision-tree-classification-python
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing

accidents = pd.read_csv('US_Accidents_Dec19.csv')


feature_cols = ['Distance(mi)','Side',
                'State','Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
                'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)','Precipitation(in)',
                'Weather_Condition','Bump', 'Crossing', 'Give_Way', 'Junction',
                'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
continuous_cols = ['Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
                'Visibility(mi)', 'Wind_Speed(mph)','Precipitation(in)']

categorical_cols = ['Side','State','Wind_Direction','Weather_Condition','Bump', 'Crossing', 'Give_Way', 'Junction',
                'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']


accidents[categorical_cols] = accidents[categorical_cols].astype('category')
categorical_df = accidents[categorical_cols]

one_hot = pd.get_dummies(categorical_df)

accidents[continuous_cols] = accidents[continuous_cols].fillna(accidents[continuous_cols].mean())

merged_features = pd.concat([accidents[continuous_cols],one_hot],axis=1)

X = merged_features
y = accidents.Severity

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
# clf = clf.fit(X, y)


#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # 68% accuracy

from sklearn.model_selection import KFold,StratifiedKFold

sfolder = StratifiedKFold(n_splits=10,random_state=0,shuffle=False)
for train, test in sfolder.split(X,y):
    clf = clf.fit(X.loc[train], y.loc[train])
    y_pred = clf.predict(X.loc[test])
    print("Accuracy:",metrics.accuracy_score(y.loc[test], y_pred))

for train, test in sfolder.split(X,y):
    m_test = test


X.loc[m_test]
