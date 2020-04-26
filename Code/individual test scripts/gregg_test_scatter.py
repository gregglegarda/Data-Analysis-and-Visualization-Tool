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

import pydotplus
import collections
from sklearn.tree import plot_tree
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
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




'''






##################################
################################### DISTANCE HISTOGRAM ##################################

data = pd.read_csv("pre_processed_data.csv")

# create scatter


#Temperature(F)	
#Wind_Chill(F)	
#Humidity(%)	
#Pressure(in)	
#Visibility(mi)	
#Wind_Direction	
#Wind_Speed(mph)	
#Precipitation(in)



#################################      SCATTER          #############################
xdata = np.arange(0,len(data),1)
ydata = data['Severity']


cmap = sns.cubehelix_palette(3, start=2.74, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
sns.set_style("white", { 'xtick.color': 'w',
 'ytick.color': 'w','axes.edgecolor': 'w','axes.labelcolor': 'w'})
color = sns.dark_palette("grey")
#plt.figure(figsize=(4, 3))

jp = sns.jointplot(x=xdata, y=ydata, kind="kde", cmap=cmap,  height = 3)
plt.gcf().set_size_inches(4, 3)


plt.savefig('dis_scat.png',
            facecolor='#1a1a1a',
            transparent=True, )
plt.close()

#################################      LOGISTIC          #############################
#######  TRAIN AND SPLIT #######
train_split = 70  # only take the first two digits since it has a %
test_split = (100 - train_split) / 100

# make table for preprocessing
column_names = ["Distance(mi)", "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
                "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
                "Severity"]

X = [list(data["Distance(mi)"]), list(data["Temperature(F)"]),
     list(data["Wind_Chill(F)"]), list(data["Humidity(%)"]), list(data["Pressure(in)"]),
     list(data["Visibility(mi)"]), list(data["Wind_Speed(mph)"]), list(data["Precipitation(in)"])]
X = np.transpose(X)
y = list(data["Severity"])

# perform PCA before splitting,
print("before PCA:",X.shape)
X = PCA(n_components=2).fit_transform(X)
print("PCA:",X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=0)
regressor = LogisticRegression(multi_class="multinomial")
regressor.fit(X_train, y_train)  # training the algorithm
y_pred = regressor.predict(X_test)
predictions = regressor.predict_proba(X_test)[:,1]

# Accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
reg_score = regressor.score(X_test, y_test)
print('Logistic Regression Model Accuracy:', round(accuracy, 2), '%.')
print('Logistic Regression Score:', round(reg_score, 2))

accuracy = round(accuracy, 2)
model_algorithm = regressor
print("train model is:", regressor)

####PLOT THE MODEL
plt.figure()
title_obj = plt.title('LOGISTIC REGRESSION')
plt.setp(title_obj, color='w')

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
y_min, y_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = regressor.predict(np.c_[xx.ravel(), yy.ravel()])


cmap = sns.cubehelix_palette(3, start=2.74, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=cmap)

# Plot also the training points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap=cmap)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.savefig('model_image.png',facecolor='#1a1a1a',transparent=True,)
plt.close()







'''
import numpy as np
gs= 3
gs = np.reshape(gs, (2,1))

print(gs.shape)
