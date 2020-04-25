import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier #https://www.datacamp.com/community/tutorials/decision-tree-classification-python
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score

try:                                                                                                                        #Following code (try/except clauses) searches for this script, and then changes the current working directory to the folder that houses it.
    start = '/Users'                                                                                                        #Code from https://stackoverflow.com/questions/43553742/finding-particular-path-in-directory-in-python
    for dirpath, dirnames, filenames in os.walk(start):
        for filename in filenames:
            if filename == "US_Accidents_Dec19.csv":
                filename = os.path.join(dirpath, filename)
                os.chdir(dirpath)
except:
    pass


try:
    start1 = "C:\\Users"
    for dirpath, dirnames, filenames in os.walk(start1):
        for filename in filenames:
            if filename == "US_Accidents_Dec19.csv":
                filename = os.path.join(dirpath, filename)
                os.chdir(dirpath)
except:
    pass

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

rf = RandomForestClassifier(n_estimators = 100, random_state = 23,verbose=3,n_jobs=-1) #https://stackoverflow.com/questions/43640546/how-to-make-randomforestclassifier-faster
rf.fit(X_train, y_train)                                                              #https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
predictions = rf.predict(X_test)# Calculate the absolute errors
print(accuracy_score(y_test,predictions))
