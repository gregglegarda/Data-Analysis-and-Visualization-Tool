import os
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import pydotplus
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import collections
import pandas as pd


class train():
    def __init__(self, train_inputs):
        super(train, self).__init__()
       #attributes
        self.accuracy = 0
        self.data = 0
        self.train_inputs = train_inputs
        self.model_algorithm = 0

        #functions
        self.data_processing()
        self.create_model()

        print("Training complete")
####==================================   CREATE MODEL FUNCTION ====================================############
    def create_model(self):
        print("Creating model...")
        print("Data in model class is:",self.train_inputs)

        #######  TRAIN AND SPLIT #######
        train_split = int(self.train_inputs[1][0] + self.train_inputs[1][1] ) #only take the first two digits since it has a %
        test_split = (100 - train_split)/100
        model_algorithim = self.train_inputs[2]

        # make table for preprocessing
        column_names = ["Start_Lat", "Start_Lng", "Distance(mi)", "Number",
                        "Temperature(F)","Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
                        "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
                        "Severity"]

        X = [list(self.data["Start_Lat"]), list(self.data["Start_Lng"]), list(self.data["Distance(mi)"]), list(self.data["Number"]),
            list(self.data["Temperature(F)"]), list(self.data["Wind_Chill(F)"]), list(self.data["Humidity(%)"]), list(self.data["Pressure(in)"]),
             list(self.data["Visibility(mi)"]), list(self.data["Wind_Speed(mph)"]), list(self.data["Precipitation(in)"])]
        X = np.transpose(X)
        y = list(self.data["Severity"])

        # data pre processing if split into 70:30
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=0)


        #########   PICKED MODEL GOES HERE   ###
        if self.train_inputs[2] == "Decision Trees":
            self.decision_tree(X_train, X_test, y_train, y_test)
        elif self.train_inputs[2] == "Random Forest":
            self.random_forest(X_train, X_test, y_train, y_test)


###================================ MACHINE LEARNING FUNCTIONS =====================================###
    def decision_tree(self, X_train, X_test, y_train, y_test):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))

        # Build decision tree model
        # visualize tree train
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        print("Model created")

        # Predict test from train
        y_pred = clf.predict(X_test)
        print("shape of y_pred:", X_test.shape)
        print("y_pred:", X_test)
        acc_score = accuracy_score(y_test, y_pred)
        self.accuracy = (acc_score * 100).round(2)
        # Accuracy
        print('DT Accuracy:', acc_score)
        print('DT Accuracy:', self.accuracy)
        self.model_algorithm = clf
        print("train model is:", clf)


    def random_forest(self, X_train, X_test, y_train, y_test):
        rf = RandomForestClassifier(n_estimators=100, random_state=23, verbose=3,
                                    n_jobs=-1)  # https://stackoverflow.com/questions/43640546/how-to-make-randomforestclassifier-faster
        rf.fit(X_train, y_train)  # https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
        predictions = rf.predict(X_test)  # Calculate the absolute errors
        errors = abs(predictions - y_test)  # Print out the mean absolute error (mae)
        # Calculate mean absolute percentage error (MAPE)
        mape = 100 * (errors / y_test)  # Calculate and display accuracy

        # Accuracy
        accuracy = 100 - np.mean(mape)
        print('Random Forest Model Accuracy:', round(accuracy, 2), '%.')
        self.accuracy = round(accuracy, 2)
        self.model_algorithm = rf
        print("train model is:", rf)


#########-------------------------------------- DATA PROCESSING AND ANALYSIS -------------------------------------- #########
    def data_processing(self):
        ######### DATA PROCESSING  #########
        print("--------------------------DATA PROCESSING--------------------------")
        print("Processing sample size of:", self.train_inputs[0])

        datafile = "US_Accidents_Dec19.csv"

        try:
            import pre_process
        except:
            print("import exception")

        data_instance = pre_process.data_frame(datafile, self.train_inputs)
        self.data = data_instance.create_dataframe()
        data_instance.cleanup_data()

        ######### DATA ANALYSIS  #########
        print("--------------------------DATA ANALYSIS--------------------------")
        try:
            import eda_stats
        except:
            print("import exception")
        data_analysis = eda_stats.eda(self.data)
        data_analysis.perform_eda()
###============================== GET DATA FUNCTIONS ===============================###########
    def get_map_data_points(self):
        return self.data
    def get_model_accuracy(self):
        return self.accuracy
    def get_model(self):
        return self.model_algorithm
