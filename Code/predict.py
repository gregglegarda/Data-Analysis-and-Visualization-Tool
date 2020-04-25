import os
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np


class predict():
    def __init__(self, inputs, dt_model):
        super(predict, self).__init__()
        self.inputs = [float(i) for i in inputs]
        self.dt_clf = dt_model

    def predict_inputs(self):
        print("Predicting inputs...")
        print("Inputs in predict class is:",self.inputs)
        print("predict model is:",self.dt_clf )

        #reshape for predicting. single row shape will be (1, len(inputs)
        self.inputs = np.reshape(self.inputs,(1,len(self.inputs)))
        print("Inputs in predict class is:", self.inputs)
        y_pred = self.dt_clf.predict(self.inputs)
        print("Prediction severity is:", y_pred)
        return y_pred[0]

