import os
import pandas as pd
from matplotlib import pyplot as plt


class eda():
    def __init__(self, data):
        super(eda, self).__init__()
        self.data = data

    def perform_eda(self):
        print("Performing EDA...")

        self.data.describe()
        plt.title('Temperature Histogram')
        # fairly evenly distributed histogram; most accidents occur around the 60-70 degree mark
        plt.hist(self.data['Temperature(F)'],bins=50,range=[-10,120], zorder=1)
        plt.tight_layout()
        plt.savefig('temp_hist.png', facecolor = '#222222', transparent = True)