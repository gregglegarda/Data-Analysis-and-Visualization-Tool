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


        #summary
        summary = self.data.describe()
        summary.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "statistic_summary.csv")),
                         index=False,
                         header=False)
        print(summary)


        #temperature histogram
        plt.title('Temperature Histogram')
        # fairly evenly distributed histogram; most accidents occur around the 60-70 degree mark
        plt.hist(self.data['Temperature(F)'],bins=50,range=[-10,120], rwidth=0.9)

        # accident times histogram
        #self.data['Start_Time'] = self.data['Start_Time'].astype('datetime64')
        #self.data['Start_Time'] = pd.to_datetime(self.data['Start_Time'])
        # bifurcated histogram, with most accidents occuring during morning and evening rush hour
        #plt.hist(self.data['Start_Time'].dt.time, bins=100, rwidth=0.9)
        #plt.title('Accident Times')



        plt.tight_layout()
        plt.savefig('temp_hist.png', facecolor = '#222222', transparent = True)