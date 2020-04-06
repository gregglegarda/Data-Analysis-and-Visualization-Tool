import os
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime


class predict():
    def __init__(self, inputs):
        super(predict, self).__init__()
        self.inputs = inputs
        self.predict_inputs()
    def predict_inputs(self):
        print("Predicting inputs...")
        print("Inputs in predict class is:",self.inputs)


