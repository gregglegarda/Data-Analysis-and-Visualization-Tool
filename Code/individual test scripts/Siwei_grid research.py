import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier #https://www.datacamp.com/community/tutorials/decision-tree-classification-python
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
from sklearn.model_selection import KFold,StratifiedKFold               # K-fold test
import time

import smtplib
from email.mime.text import MIMEText  
from time import sleep

def sentemail(body):
    host = 'smtp.163.com'  
    port = 465  
    sender = 'stevenking2020@163.com'  
    pwd = 'QBIPBFEEKPSLGVFL'  
    receiver = 'hitcw26@163.com' 
    msg = MIMEText(body, 'html') 
    msg['subject'] = '事件通知' 
    msg['from'] = sender  
    msg['to'] = receiver  
    try:
        s = smtplib.SMTP_SSL(host, port)  
        s.login(sender, pwd)  
        s.sendmail(sender, receiver, msg.as_string())
        print ('Done.sent email success')
    except smtplib.SMTPException:
        print ('Error.sent email fail')
        
sentemail("<h1>已完成</h1>")


#import data and preprocessing
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
             
# 
accidents[categorical_cols] = accidents[categorical_cols].astype('category')
categorical_df = accidents[categorical_cols]

one_hot = pd.get_dummies(categorical_df)

accidents[continuous_cols] = accidents[continuous_cols].fillna(accidents[continuous_cols].mean())

merged_features = pd.concat([accidents[continuous_cols],one_hot],axis=1)

X = merged_features
y = accidents.Severity


# model test
def single_test(X, y, depth):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    # build model
    mdl = DecisionTreeClassifier(max_depth=depth)
    # train model
    mdl = mdl.fit(X_train,y_train)
    # predict
    y_pred = mdl.predict(X_test)
    # accuracy
    acc = metrics.accuracy_score(y_test, y_pred)
    return mdl, acc

start = time.clock()
m_mdl, m_acc = single_test(X, y, None)
print("tree depth:", m_mdl.get_depth(), "accuracy:", m_acc)
print(time.clock() - start, "seconds used")

start = time.clock()
m_mdl, m_acc = single_test(X, y, 80)
print("tree depth:", m_mdl.get_depth(), "accuracy:", m_acc)
print(time.clock() - start, "seconds used")

start = time.clock()
m_mdl, m_acc = single_test(X, y, 10)
print("tree depth:", m_mdl.get_depth(), "accuracy:", m_acc)
print(time.clock() - start, "seconds used")


# k fold test
def K_fold_eval(mdl, K):
    sfolder = StratifiedKFold(n_splits=K, random_state=1,shuffle=True)
    tot = 0
    for train, test in sfolder.split(X,y):
        mdl = mdl.fit(X.loc[train], y.loc[train])
        y_pred = clf.predict(X.loc[test])
        acc = metrics.accuracy_score(y.loc[test], y_pred)
        tot = tot + acc
    return tot/K
    
mdl = DecisionTreeClassifier(max_depth=depth)
K_fold_eval(clf, 10)


# grid research
# for model DecisionTreeClassifier, parameters are below:
# criterion, ["gini", "entropy"]
# spliter, ["best", "random"]
# max_depth,
# min_smaples_split， seems not important
start = time.clock()
acc_li = []
for cri in ["gini", "entropy"]:
    for spl in ["best", "random"]:
        for mxd in range(5, 100, 4):
            mdl = DecisionTreeClassifier(criterion=cri, splitter=spl, max_depth=mxd)
            m_acc = K_fold_eval(mdl, 10)
            acc_li.append(m_acc)

print(time.clock() - start, "seconds used")

sentemail("<h1>已完成</h1>")
