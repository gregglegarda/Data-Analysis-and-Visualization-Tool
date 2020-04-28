import os
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA
from PyQt5.QtWidgets import  QMessageBox


class predict():
    def __init__(self, inputs, dt_model, model_name, model_name2, pca_model):
        super(predict, self).__init__()
        self.inputs = [float(i[:-4]) for i in inputs]
        self.dt_clf = dt_model
        self.model_name = model_name ## from the model display
        self.model_name2 = model_name2.upper()  ## from the algorthihm drop box
        self.pca_model = pca_model

    def predict_inputs(self):
        print("Predicting inputs...")
        print("Inputs in predict class is:",self.inputs)
        print("predict model is:",self.dt_clf )

        # ======= PERFORM PCA ==========#
        # for KNN and Logistic Regression
        print("model1", self.model_name)
        print("model2", self.model_name2)
        if (self.model_name ==  self.model_name2 == "KNN") or (self.model_name == self.model_name2 == "LOGISTIC REGRESSION"):
            self.inputs = np.reshape(self.inputs, (1, -1))
            print("shape of inputs\ before pca\n", self.inputs.shape)
            self.inputs = self.pca_model.transform(self.inputs)
            print("PCA PERFORMED")

            # reshape for predicting. single row shape will be (1, len(inputs)
            self.inputs = np.reshape(self.inputs, (1, -1))
            print("Inputs in predict class is:", self.inputs)
            y_pred = self.dt_clf.predict(self.inputs)
            print("Prediction severity is:", y_pred)
            return y_pred[0]
        elif self.model_name ==  self.model_name2:

            #reshape for predicting. single row shape will be (1, len(inputs)
            self.inputs = np.reshape(self.inputs,(1,-1))
            print("Inputs in predict class is:", self.inputs)
            y_pred = self.dt_clf.predict(self.inputs)
            print("Prediction severity is:", y_pred)
            return y_pred[0]
        else:

            msg = QMessageBox()
            msg.setText("Please train your new chosen model")
            msg.exec_()

