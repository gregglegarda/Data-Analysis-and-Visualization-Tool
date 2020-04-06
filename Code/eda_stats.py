import os
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime


class eda():
    def __init__(self, data):
        super(eda, self).__init__()
        self.data = data

    def perform_eda(self):
        print("Performing EDA...")


        ################################### SUMMARY OF STATS ##################################
        summary = self.data.describe()
        summary = summary.T
        summary.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "statistic_summary.csv")), index=False, header=False)
        print(summary)


        ################################### TEMPERATURE HISTOGRAM ##################################
        # fairly evenly distributed histogram; most accidents occur around the 60-70 degree mark
        plt.figure(figsize=(6, 4))
        plt.hist(self.data['Temperature(F)'],bins=50,range=[-10,120], rwidth=0.9)

        #line colors
        title_obj = plt.title('Temperature Histogram')
        plt.setp(title_obj, color='w')
        plt.tick_params(axis='both', colors='white')
        ax = plt.gca()
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.spines['left'].set_color('w')

        plt.tight_layout()
        plt.savefig('temp_hist.png',
                    facecolor = '#1a1a1a',
                    transparent = True,)