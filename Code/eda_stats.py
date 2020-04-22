import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
from PIL import Image



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







        ################################### DISTANCE HISTOGRAM ##################################

        # create histogram
        plt.figure(figsize=(4, 3))
        plt.hist(self.data['Distance(mi)'], bins=10, range=[0,0.02], edgecolor='darkgray',
                 linewidth=0.5)  # color="gray",)

        # figure settings
        title_obj = plt.title('Distance Histogram')
        plt.setp(title_obj, color='w')
        plt.tick_params(axis='both', colors='white')
        ax = plt.gca()
        # ax.locator_params(axis='x', integer=True)
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.spines['left'].set_color('w')
        plt.tight_layout()

        # save hist
        plt.savefig('dis_hist.png',
                    facecolor='#1a1a1a',
                    transparent=True, )



        ################################### TEMPERATURE HISTOGRAM ##################################
        # fairly evenly distributed histogram; most accidents occur around the 60-70 degree mark

        #create histogram
        plt.figure(figsize=(4, 3))
        plt.hist(self.data['Temperature(F)'],bins=30, range=[-9,121],  edgecolor='darkgray', linewidth=0.5) #color="gray",)

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


        ################################### WIND CHILL HISTOGRAM ##################################

        # create histogram
        plt.figure(figsize=(4, 3))
        plt.hist(self.data['Wind_Chill(F)'], bins=30, edgecolor='darkgray',
                 linewidth=0.5,  range=[24, 76], align='mid') # color="gray",)

        # figure settings
        title_obj = plt.title('Wind Chill Histogram')
        plt.setp(title_obj, color='w')
        plt.tick_params(axis='both', colors='white')
        ax = plt.gca()
        #ax.locator_params(axis='x', integer=True)
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.spines['left'].set_color('w')
        plt.tight_layout()

        # save hist
        plt.savefig('wch_hist.png',
                    facecolor='#1a1a1a',
                    transparent=True, )



        ################################### HUMIDITY HISTOGRAM ##################################

        # create histogram
        plt.figure(figsize=(4, 3))
        plt.hist(self.data['Humidity(%)'], bins=30, edgecolor='darkgray',
                 linewidth=0.5,  range=[-1, 101]) # color="gray",)

        # figure settings
        title_obj = plt.title('Humidity Histogram')
        plt.setp(title_obj, color='w')
        plt.tick_params(axis='both', colors='white')
        ax = plt.gca()
        #ax.locator_params(axis='x', integer=True)
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.spines['left'].set_color('w')
        plt.tight_layout()

        # save hist
        plt.savefig('hum_hist.png',
                    facecolor='#1a1a1a',
                    transparent=True, )

        ################################### PRESSURE HISTOGRAM ##################################

        # create histogram
        plt.figure(figsize=(4, 3))
        plt.hist(self.data['Pressure(in)'], bins=10, range=[28,32],edgecolor='darkgray',
                 linewidth=0.5)  # color="gray",)

        # figure settings
        title_obj = plt.title('Pressure Histogram')
        plt.setp(title_obj, color='w')
        plt.tick_params(axis='both', colors='white')
        ax = plt.gca()
        # ax.locator_params(axis='x', integer=True)
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.spines['left'].set_color('w')
        plt.tight_layout()

        # save hist
        plt.savefig('prs_hist.png',
                    facecolor='#1a1a1a',
                    transparent=True, )

        ################################### VISIBILITY HISTOGRAM ##################################

        # create histogram
        plt.figure(figsize=(4, 3))
        plt.hist(self.data['Visibility(mi)'], bins=np.arange(-1,11)+0.5, edgecolor='darkgray',
                 linewidth=0.5)  # color="gray",)

        # figure settings
        title_obj = plt.title('Visibility Histogram')
        plt.setp(title_obj, color='w')
        plt.tick_params(axis='both', colors='white')
        ax = plt.gca()
        # ax.locator_params(axis='x', integer=True)
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.spines['left'].set_color('w')
        plt.tight_layout()

        # save hist
        plt.savefig('vis_hist.png',
                    facecolor='#1a1a1a',
                    transparent=True, )

        ################################### WIND SPEED HISTOGRAM ##################################

        # create histogram
        plt.figure(figsize=(4, 3))
        plt.hist(self.data['Wind_Speed(mph)'], bins=20, edgecolor='darkgray',
                 linewidth=0.5, range=[0, 21])  # color="gray",)

        # figure settings
        title_obj = plt.title('Wind Speed Histogram')
        plt.setp(title_obj, color='w')
        plt.tick_params(axis='both', colors='white')
        ax = plt.gca()
        # ax.locator_params(axis='x', integer=True)
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.spines['left'].set_color('w')
        plt.tight_layout()

        # save hist
        plt.savefig('wsp_hist.png',
                    facecolor='#1a1a1a',
                    transparent=True, )

        ################################### PRECIPITATION HISTOGRAM ##################################

        # create histogram
        plt.figure(figsize=(4, 3))
        plt.hist(self.data['Precipitation(in)'], bins=10, range = [0,0.5], edgecolor='darkgray',
                 linewidth=0.5 ) # color="gray",)

        # figure settings
        title_obj = plt.title('Precipitation Histogram')
        plt.setp(title_obj, color='w')
        plt.tick_params(axis='both', colors='white')
        ax = plt.gca()
        # ax.locator_params(axis='x', integer=True)
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.spines['left'].set_color('w')
        plt.tight_layout()

        # save hist
        plt.savefig('prc_hist.png',
                    facecolor='#1a1a1a',
                    transparent=True, )

        ################################### SEVERITY HISTOGRAM ##################################

        # create histogram
        plt.figure(figsize=(4, 3))
        plt.hist(self.data['Severity'], bins=np.arange(0, 5) + 0.5, ec="k", edgecolor='darkgray',
                 linewidth=0.5)  # range=[1, 4], # color="gray",)

        # figure settings
        title_obj = plt.title('Severity Histogram')
        plt.setp(title_obj, color='w')
        plt.tick_params(axis='both', colors='white')
        ax = plt.gca()
        ax.locator_params(axis='x', integer=True)
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.spines['left'].set_color('w')
        plt.tight_layout()

        # save hist
        plt.savefig('sev_hist.png',
                    facecolor='#1a1a1a',
                    transparent=True, )

        ################################### Concatenate histograms ##################################
        #concatenate
        im1 = Image.open('sev_hist.png')
        im2 = Image.open('dis_hist.png')
        im3 = Image.open('temp_hist.png')
        im4 = Image.open('wch_hist.png')
        im5 = Image.open('hum_hist.png')
        im6 = Image.open('prs_hist.png')
        im7 = Image.open('vis_hist.png')
        im8 = Image.open('wsp_hist.png')
        im9 = Image.open('prc_hist.png')
        self.get_concat_v_multi_resize([im1, im2, im3, im4, im5, im6, im7, im8, im9]).save('histograms.png')

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

    def get_concat_v_multi_resize(self, im_list, resample=Image.BICUBIC):
        min_width = min(im.width for im in im_list)
        im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)), resample=resample)
                          for im in im_list]
        total_height = sum(im.height for im in im_list_resize)
        dst = Image.new('RGB', (min_width, total_height))
        pos_y = 0
        for im in im_list_resize:
            dst.paste(im, (0, pos_y))
            pos_y += im.height
        return dst