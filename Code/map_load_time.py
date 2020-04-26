import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy, scipy, matplotlib
import matplotlib.ticker as tick
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
import math

import numpy as np
import csv


class map_load_time():
    def __init__(self):
        super(map_load_time, self).__init__()
        self.df_csv_load_times = 0
    def calculate_load_time(self, num_samples):
        #make csv file for load time and put data points for every run.

        try:
            #try reading first
            open('map_load_time.csv', mode='r')

        except:
            # save data points to make it better overtime
            with open('map_load_time.csv', mode='w') as employee_file:
                employee_writer = csv.writer(employee_file)
                employee_writer.writerow(["x", "y"])
                employee_writer.writerow([1, 1])
                employee_writer.writerow([100, 3])
                employee_writer.writerow([1000, 4])
                employee_writer.writerow([5000, 6])
                employee_writer.writerow([10000, 5])
                employee_writer.writerow([50000, 9])
                employee_writer.writerow([100000, 15])
                employee_writer.writerow([150000, 31])
                employee_writer.writerow([200000, 50])
                employee_writer.writerow([250000, 60])
                employee_writer.writerow([300000, 80])
                employee_writer.writerow([350000, 100])
                employee_writer.writerow([400000, 140])
                employee_writer.writerow([450000, 160])
                employee_writer.writerow([500000, 240])
                employee_writer.writerow([550000, 275])
                employee_writer.writerow([600000, 290])
                employee_writer.writerow([800000, 350])
                employee_writer.writerow([1000000,360])
            employee_file.close()


        #make the data frame
        filename = os.path.expanduser(os.path.abspath(os.path.join(os.path.dirname(__file__), "map_load_time.csv")))

        data = pd.read_csv(filename)
        data = data.sort_values(by='x')
        self.df_csv_load_times = data.copy()
        print(data)
        x1 = np.array(data.iloc[:, 0])
        y1 = np.array(data.iloc[:, 1])
        x = x1.reshape(-1, 1)
        y = y1.reshape(-1, 1)
        print("x is:\n", x)
        print("y is:\n",y)


        #CURVED FITTING SIGMOID
        print("lenght of indexes in csv",len(data.index) )
        print("x1 and y1v", x1, y1)
        y_predict_point =  self.fit_the_curve(len(data.index), x1,y1, num_samples)
        print("geo map prediction for ", num_samples," is ", y_predict_point)


        #if new data point is not in the csv file, save.
        print("Sample number Not in CSV:", num_samples not in data.x.values)
        if num_samples not in data.x.values:
            # save data points to make it better overtime
            with open('map_load_time.csv', mode='a') as employee_file:
                employee_writer = csv.writer(employee_file)
                employee_writer.writerow([num_samples, y_predict_point])
            employee_file.close()

        #else, if its there, take the higher value
        #else:
            #if y_predict_point >
            #with open('map_load_time.csv', mode='a') as employee_file:
                #employee_writer = csv.writer(employee_file)
                #employee_writer.writerow([num_samples, y_predict_point])
            #employee_file.close()

        return y_predict_point

    def reformat_large_tick_values(self, tick_val, pos):
        """
        Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).
        """
        if tick_val >= 1000000000:
            val = round(tick_val / 1000000000, 1)
            new_tick_format = '{:}B'.format(val)
        elif tick_val >= 1000000:
            val = round(tick_val / 1000000, 1)
            new_tick_format = '{:}M'.format(val)
        elif tick_val >= 1000:
            val = round(tick_val / 1000, 1)
            new_tick_format = '{:}K'.format(val)
        elif tick_val < 1000:
            new_tick_format = round(tick_val, 1)
        else:
            new_tick_format = tick_val

        # make new_tick_format into a string value
        new_tick_format = str(new_tick_format)

        # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
        index_of_decimal = new_tick_format.find(".")

        if index_of_decimal != -1:
            value_after_decimal = new_tick_format[index_of_decimal + 1]
            if value_after_decimal == "0":
                # remove the 0 after the decimal point since it's not needed
                new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal + 2:]

        return new_tick_format

    def sigmoid(self,x, k, x0):
        max_secs = float(self.df_csv_load_times["y"].max())
        print("max secs", max_secs)
        return max_secs / (1 + np.exp(-k * (x - x0)))
    def fit_the_curve(self, n_csv_samples,x1, y1, n_samples):
        max_sample_num = float(self.df_csv_load_times["x"].max())
        print("max sample num", max_sample_num)
        # fit_the_curve(len(data.index), x1,y1)
        popt, pcov = curve_fit(self.sigmoid, x1, y1, method='dogbox', bounds=([0., 100.], [0.00001, max_sample_num]))


        # Parameters of the true function

        # Build the true function and fit the data
        x = np.linspace(0, int(max_sample_num), num=n_csv_samples)
        #fit the data in the true function
        y_fitted = self.sigmoid(x, *popt)
        #predict one sample
        y_predict_point = self.sigmoid(n_samples, *popt)

        ################################### MAP LOAD TIME CHART ##################################
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y_fitted, '--', label='predict', color= '#1f77b4')
        ax.plot(x1, y1, 'o', label='seconds', color= '#1f77b4')
        ax.plot(n_samples, y_predict_point, 'o', label='current', color='red')
        ax.legend()

        # figure settings
        title_obj = plt.title('GEO MAP LOAD TIMES')
        legend = plt.legend(facecolor="#1a1a1a")
        plt.setp(legend.get_texts(), color='w')
        legend
        plt.setp(title_obj, color='w')
        plt.tick_params(axis='both', colors='white')
        ax = plt.gca()
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.spines['left'].set_color('w')
        ax.xaxis.set_major_formatter(tick.FuncFormatter(self.reformat_large_tick_values))
        plt.tight_layout()

        # save hist
        plt.savefig('map_load_time.png',
                    facecolor='#1a1a1a',
                    transparent=True, )
        plt.close()
        return y_predict_point





