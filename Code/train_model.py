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
import collections
import pandas as pd


class train():
    def __init__(self, train_inputs):
        super(train, self).__init__()
        self.accuracy = 0
        self.data = 0
        self.train_inputs = train_inputs

        self.data_processing()
        self.create_model()

        print("Training complete")
    def create_model(self):
        print("Creating model...")
        print("Data in model class is:",self.train_inputs)

        #######=================  PUT TRAINING ALGORITHM HERE   =================#######
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
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))

        #Build decision tree model
        # visualize tree train
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        print("Model created")

        # Predict test from train
        y_pred = clf.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        self.accuracy = (acc_score*100).round(2)
        # Accuracy
        print('Accuracy:', acc_score)
        print('Accuracy:', self.accuracy)




    def data_processing(self):
        #########-------------------------------------- DATA PROCESSING -------------------------------------- #########
        print("--------------------------DATA PROCESSING--------------------------")
        print("Processing sample size of:", self.train_inputs[0])

        datafile = "US_Accidents_Dec19.csv"

        try:
            from Code import pre_process
        except:
            print("import exception")

        data_instance = pre_process.data_frame(datafile, self.train_inputs)
        self.data = data_instance.create_dataframe()
        data_instance.cleanup_data()

        #########-------------------------------------- DATA ANALYSIS -------------------------------------- #########
        print("--------------------------DATA ANALYSIS--------------------------")
        try:
            import eda_stats
        except:
            print("import exception")
        data_analysis = eda_stats.eda(self.data)
        data_analysis.perform_eda()

    def get_map_data_points(self):
        return self.data
    def get_model_accuracy(self):
        return self.accuracy