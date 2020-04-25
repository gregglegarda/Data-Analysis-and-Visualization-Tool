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
    def __init__(self, data, app):
        super(eda, self).__init__()
        self.data = data
        self.app = app
        self.sns_data = data.copy()


    def perform_eda(self):
        print("Performing EDA...")


        ################################### SUMMARY OF STATS ##################################
        summary = self.data.describe()
        summary = (summary.T).round(4)
        summary.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "statistic_summary.csv")), index=False, header=False)
        print(summary)


        ################################### DISTANCE HISTOGRAM ##################################

        self.create_histo('Distance(mi)', [0,0.02], 10, 'Distance Histogram', 'dis_hist.png')
        self.create_scatter('Distance Scatterplot', 'Distance(mi)', 'dis_scat.png')


        ################################### TEMPERATURE HISTOGRAM ##################################

        self.create_histo('Temperature(F)', [-9,121], 30,  'Temperature Histogram', 'temp_hist.png')
        self.create_scatter('Temperature Scatterplot', 'Temperature(F)', 'temp_scat.png')


        ################################### WIND CHILL HISTOGRAM ##################################

        self.create_histo('Wind_Chill(F)', [24, 76], 30, 'Wind Chill Histogram', 'wch_hist.png')
        self.create_scatter('Wind Chill Scatterplot', 'Wind_Chill(F)', 'wch_scat.png')


        ################################### HUMIDITY HISTOGRAM ##################################

        self.create_histo('Humidity(%)', [-1, 101], 30, 'Humidity Histogram', 'hum_hist.png')
        self.create_scatter('Humidity Scatterplot', 'Humidity(%)', 'hum_scat.png')



        ################################### PRESSURE HISTOGRAM ##################################

        self.create_histo('Pressure(in)', [28,32], 10, 'Pressure Histogram', 'prs_hist.png')
        self.create_scatter('Pressure Scatterplot', 'Pressure(in)', 'prs_scat.png')


        ################################### VISIBILITY HISTOGRAM ##################################

        self.create_histo('Visibility(mi)', None, np.arange(-1,11)+0.5, 'Visibility Histogram', 'vis_hist.png', )
        self.create_scatter('Visibility Scatterplot', 'Visibility(mi)', 'vis_scat.png')



        ################################### WIND SPEED HISTOGRAM ##################################

        self.create_histo('Wind_Speed(mph)', [0, 21], 20, 'Wind Speed Histogram',
                            'wsp_hist.png', )
        self.create_scatter('Wind Speed Scatterplot', 'Wind_Speed(mph)', 'wsp_scat.png')



        ################################### PRECIPITATION HISTOGRAM ##################################
        self.create_histo('Precipitation(in)', [0,0.5], 10, 'Precipitation Histogram',
                            'prc_hist.png', )
        self.create_scatter('Precipitation Scatterplot', 'Precipitation(in)', 'prc_scat.png')


        ################################### SEVERITY HISTOGRAM ##################################
        self.create_histo('Severity', None, np.arange(0, 5) + 0.5, 'Severity Histogram',
                            'sev_hist.png', )
        self.create_scatter('Severity Scatterplot', 'Severity', 'sev_scat.png')



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




        ################################### Concatenate scatterplots ##################################
        #concatenate
        im1s = Image.open('sev_scat.png')
        im2s = Image.open('dis_scat.png')
        im3s = Image.open('temp_scat.png')
        im4s = Image.open('wch_scat.png')
        im5s = Image.open('hum_scat.png')
        im6s = Image.open('prs_scat.png')
        im7s = Image.open('vis_scat.png')
        im8s = Image.open('wsp_scat.png')
        im9s = Image.open('prc_scat.png')
        self.get_concat_v_multi_resize([im1s, im2s, im3s, im4s, im5s, im6s, im7s, im8s, im9s]).save('scatterplots.png')

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
        plt.figure()
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
        plt.close()

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

    def create_histo(self, column, range, bins,   hist_title, png_title,):
        # create histogram
        plt.figure(figsize=(4, 3))
        plt.hist(self.data[column], bins=bins, edgecolor='darkgray',
                 linewidth=0.5, range=range, align='mid')  # color="gray",)

        # figure settings
        title_obj = plt.title(hist_title)
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
        plt.savefig(png_title,
                    facecolor='#1a1a1a',
                    transparent=True )
        plt.close()
    def create_scatter(self, scat_title, column, png_title ):
        x = np.arange(0, len(self.data), 1)
        y = self.data[column]

        if len(self.data) >=5000:
            self.create_scatter_heatmap(column, png_title)
        else:
            # make and save scatter plot
            plt.figure(figsize=(4, 3))
            plt.title(scat_title)
            plt.scatter(x, y, edgecolor='darkgray')  # color="gray",)

            # figure settings
            title_obj = plt.title(scat_title)
            plt.setp(title_obj, color='w')
            plt.tick_params(axis='both', colors='white')
            ax = plt.gca()
            # ax.locator_params(axis='x', integer=True)
            ax.spines['bottom'].set_color('w')
            ax.spines['top'].set_color('w')
            ax.spines['right'].set_color('w')
            ax.spines['left'].set_color('w')
            plt.tight_layout()


            #save scatter
            plt.savefig(png_title,
                        facecolor='#1a1a1a',
                        transparent=True )
            plt.close()
            self.app.processEvents()

    def create_scatter_heatmap(self, column, png_title):
        xdata = np.arange(0, len(self.data), 1)
        ydata = self.data[column]

        #import mylib

        #a4_dims = (11.7, 8.27)
        #df = mylib.load_data()
        #fig, ax = plt.subplots(figsize=a4_dims)
        #sns.violinplot(ax=ax, data=df, **violin_options)

        plt.figure()
        cmap = sns.cubehelix_palette(3, start=2.74, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
        sns.set_style("white", {'xtick.color': 'w',
                                'ytick.color': 'w', 'axes.edgecolor': 'w', 'axes.labelcolor': 'w'})

        sns.jointplot(x=xdata, y=ydata, kind="kde", cmap=cmap, height = 3)
        plt.gcf().set_size_inches(4, 3)

        # save scatter
        plt.savefig(png_title,
                    facecolor='#1a1a1a',
                    transparent=True )
        plt.close()
        self.app.processEvents()