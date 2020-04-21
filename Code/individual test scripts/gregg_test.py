import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
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



data = pd.read_csv('pre_processed_data.csv')
'''

Severity	
Start_Time	
End_Time	
Start_Lat	
Start_Lng	
Distance(mi)	

Description	
Number	
Street	
Side	
City	
County	
State	
Zipcode	

Timezone	
Airport_Code	
Weather_Timestamp	

-------------------WEATHER CONDITIONS----------------
Temperature(F)	
Wind_Chill(F)	
Humidity(%)	
Pressure(in)	
Visibility(mi)	
Wind_Direction	
Wind_Speed(mph)	
Precipitation(in)	
Weather_Condition - Clear, Overcast, Partly Cloudy, Scattered Clouds, Mostly Cloudy, Fair, Wintry Mix


------------------STREET CONDITIONS---------------------
Amenity	Bump	
Crossing	
Give_Way	
Junction	
No_Exit	Railway	
Roundabout	
Station	Stop	
Traffic_Calming	
Traffic_Signal	
Turning_Loop	

--------------- TIMEOF DAY CONDITIONS-------------
Sunrise_Sunset	
Civil_Twilight	
Nautical_Twilight	
Astronomical_Twilight


'''







def sigmoid(x, k, x0):
    return 360.0 / (1 + np.exp(-k * (x - x0)))


def fit_the_curve1(num_samples, data_x, data_y):
    # Parameters of the true function
    n_samples = num_samples
    true_x0 = 1000000 / 2
    true_k = 360
    sigma = 0.2 * true_k

    # Build the true function and add some noise
    x = np.linspace(0, 1000000, num=n_samples)
    y = sigmoid(x, k=true_k, x0=true_x0)
    y_with_noise = y + sigma * np.random.randn(n_samples)

    # Sample the data from the real function (this will be your data)
    #some_points = np.random.choice(360, size=30)  # take 30 data points
    # print("some points", some_points)
    xdata = data_x
    ydata = data_y

    # Fit the curve
    popt, pcov = curve_fit(sigmoid, xdata, ydata)
    estimated_k, estimated_x0 = popt

    # Plot the fitted curve
    y_fitted = sigmoid(x, k=estimated_k, x0=estimated_x0)

    # Plot everything for illustration
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y_fitted, '--', label='fitted')
    ax.plot(x, y, '-', label='true')
    ax.plot(xdata, ydata, 'o', label='samples')
    ax.legend()

    plt.savefig('sample.png',
                facecolor='#1a1a1a',
                transparent=True, )




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


#CURVED FITTING SIGMOID
print("lenght of indexes in csv",len(data.index) )
print("x1 and y1v", x1, y1)

#fit_the_curve(len(data.index), x1,y1)
popt, pcov = curve_fit(sigmoid, x1, y1, method='dogbox', bounds= ([0.,100.],[0.00001,1000000.]))


# Build the true function and add some noise
n_samples = 19
true_x0 = 1000000 / 2
true_k = 360
sigma = 0.2 * true_k
x = np.linspace(0, 1000000, num=n_samples)
y_fitted = sigmoid(x, *popt)

# Plot the fitted curve

# Plot everything for illustration
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y_fitted, '--', label='fitted')
ax.plot(x1, y1, 'o', label='samples')
ax.legend()

plt.savefig('sample.png',
            facecolor='#1a1a1a',
            transparent=True, )
################################### TEST  ##################################
