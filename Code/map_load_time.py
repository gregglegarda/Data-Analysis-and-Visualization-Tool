import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy, scipy, matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings

import numpy as np
import csv


class map_load_time():
    def __init__(self):
        super(map_load_time, self).__init__()
        self.x = 0
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
                employee_writer.writerow([10000, 5])
                employee_writer.writerow([50000, 8])
                employee_writer.writerow([100000, 15])
                employee_writer.writerow([200000, 50])
                employee_writer.writerow([300000, 80])
                employee_writer.writerow([400000, 140])
                employee_writer.writerow([500000, 240])
                employee_writer.writerow([600000, 290])
                employee_writer.writerow([1000000,360])
            employee_file.close()


        #make the data frame
        filename = os.path.expanduser(os.path.abspath(os.path.join(os.path.dirname(__file__), "map_load_time.csv")))
        data = pd.read_csv(filename)
        data = data.sort_values(by='x')
        print(data)
        x1 = np.array(data.iloc[:, 0])
        y1 = np.array(data.iloc[:, 1])
        x = x1.reshape(-1, 1)
        y = y1.reshape(-1, 1)
        print("x is:\n", x)
        print("y is:\n",y)


        # SVC MODEL
        from sklearn.svm import SVC
        y=y1
        model = SVC(kernel='poly')
        model.fit(x, y)
        y_pred = model.predict(x)
        # model prediction
        x1 = np.array(num_samples).reshape(-1, 1)
        y_predict_point = float(model.predict(x1)[0])
        print("model prediction from ", num_samples, " number of samples:\n", y_predict_point)

        # train the model to predict the linear regression line
        #from sklearn.linear_model import LinearRegression
        #model = LinearRegression()
        #model.fit(x, y)  # training the algorithm
        #y_pred = model.predict(x)
        # model prediction
        #x1 = np.array(num_samples).reshape(-1, 1)
        #y_predict_point = float(model.predict(x1)[0])
        #print("model prediction from ", num_samples, " number of samples:\n", y_predict_point)


        #curved model alternate to linear regression
        #x=x1
        #y=y1
        #xmult, yinter = np.polyfit(x,np.log(y),1)
        #y_predict_point = np.exp(xmult * num_samples) * np.exp(yinter)
        #print("model prediction from ", num_samples, " number of samples:\n", y_predict_point)
        #y_pred = np.exp(xmult * x) * np.exp(yinter)


        #if new data point is in the csv file, do not save.


        # save data points to make it better overtime
        with open('map_load_time.csv', mode='a') as employee_file:
            employee_writer = csv.writer(employee_file)
            employee_writer.writerow([num_samples, y_predict_point])
        employee_file.close()



        ################################### MAP LOAD TIME CHART ##################################

        # create histogram
        plt.figure(figsize=(4, 3))
        #xline = np.array(np.linspace(100, 500000, len(y_pred))).reshape(-1, 1)
        plt.plot(x, y_pred)
        plt.scatter(x, y)
        print("x is:\n", x)
        print("y is:\n", y)
        plt.scatter(num_samples, y_predict_point, color="red")


        # figure settings
        title_obj = plt.title('GEO MAP LOAD TIMES')
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
        plt.savefig('map_load_time.png',
                    facecolor='#1a1a1a',
                    transparent=True, )

        return y_predict_point