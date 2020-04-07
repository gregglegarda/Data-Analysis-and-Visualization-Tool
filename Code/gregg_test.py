import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np



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

Temperature(F)	
Wind_Chill(F)	
Humidity(%)	
Pressure(in)	
Visibility(mi)	
Wind_Direction	
Wind_Speed(mph)	
Precipitation(in)	
Weather_Condition - Clear, Overcast, Partly Cloudy, Scattered Clouds, Mostly Cloudy, Fair, Wintry Mix
	
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
Sunrise_Sunset	
Civil_Twilight	
Nautical_Twilight	
Astronomical_Twilight


'''

################################### TEMPERATURE HISTOGRAM ##################################
# fairly evenly distributed histogram; most accidents occur around the 60-70 degree mark
#plt.figure(figsize=(8, 6))
plt.hist(data['Temperature(F)'],bins=50,range=[-10,120], rwidth=0.9)

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
#plt.savefig('gregg_output.png',
            #facecolor = '#1a1a1a',
            #transparent = True,
            #)

print("first",data['Severity'].mean())
print("second",data['Severity'])
data['Severity'] = data['Severity'].fillna(0)
print("third",data['Severity'])






################################### TEMPERATURE HISTOGRAM ##################################
# calculate the correlation matrix

sns.set()

ax = sns.heatmap(data)

# Saving image
#plt.savefig('heatmap-02.png')

# In case you dont want to save image but just displya it
plt.show()