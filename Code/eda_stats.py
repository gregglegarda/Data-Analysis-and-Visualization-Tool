import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl


class eda():
    def __init__(self, data):
        super(eda, self).__init__()
        self.data = data
        self.sns_data = data.copy()

    def perform_eda(self):
        print("Performing EDA...")


        ################################### SUMMARY OF STATS ##################################
        summary = self.data.describe()
        summary = (summary.T).round(4)
        summary.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "statistic_summary.csv")), index=False, header=False)
        print(summary)


        ################################### TEMPERATURE HISTOGRAM ##################################
        # fairly evenly distributed histogram; most accidents occur around the 60-70 degree mark

        #create histogram
        plt.figure(figsize=(4, 3))
        plt.hist(self.data['Temperature(F)'],bins=30, range=[-10,120],  edgecolor='darkgray', linewidth=0.5) #color="gray",)

        #figure settings
        title_obj = plt.title('Temperature Histogram')
        plt.setp(title_obj, color='w')
        plt.tick_params(axis='both', colors='white')
        ax = plt.gca()
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.spines['left'].set_color('w')
        plt.tight_layout()

        #save hist
        plt.savefig('temp_hist.png',
                    facecolor = '#1a1a1a',
                    transparent = True,)

        ################################### HEAT MAP ##################################


        del self.sns_data['Start_Time']
        del self.sns_data['End_Time']
        del self.sns_data['Description']
        del self.sns_data['Street']
        del self.sns_data['Side']
        del self.sns_data['City']
        del self.sns_data['County']
        del self.sns_data['State']
        del self.sns_data['Zipcode']
        del self.sns_data['Sunrise_Sunset']
        del self.sns_data['Civil_Twilight']
        del self.sns_data['Nautical_Twilight']
        del self.sns_data['Astronomical_Twilight']

        del self.sns_data['Amenity']
        del self.sns_data['Bump']
        del self.sns_data['Crossing']
        del self.sns_data['Give_Way']
        del self.sns_data['Junction']
        del self.sns_data['No_Exit']
        del self.sns_data['Railway']
        del self.sns_data['Roundabout']
        del self.sns_data['Station']
        del self.sns_data['Stop']
        del self.sns_data['Traffic_Calming']
        del self.sns_data['Traffic_Signal']
        del self.sns_data['Turning_Loop']

        #rename columns
        self.sns_data.columns = ["sev","lat","lng","dis","num","tmp","wch","hum","prs","vis","wsp","prc"]
        print("info of data:", self.data.info())
        print("info of sns:", self.sns_data.info())

        #create correlation matrix
        corrmatrix = self.sns_data.corr()
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        heatplot = ax.imshow(corrmatrix, cmap='PuBu')

        print("corrmatrix:\n",corrmatrix)
        #heatmap settings
        ax.set_xticklabels(corrmatrix.columns)
        ax.set_yticklabels(corrmatrix.index)


        ax.set_xticks(np.arange(len(self.sns_data.columns)))
        ax.set_yticks(np.arange(len(self.sns_data.columns)))
        ax.set_title('Correlation Matrix', color = "white")
        plt.tick_params(axis='both', colors='white', labelsize = 6)
        #ax = plt.gca()
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.spines['left'].set_color('w')
        plt.xticks(rotation=0)
        cbar = plt.colorbar(heatplot)
        cbar.ax.tick_params(labelsize=6, colors='white' )
        #save corr matrix
        plt.savefig('correlation_matrix.png',
                    facecolor='#1a1a1a',
                    transparent=True, )
