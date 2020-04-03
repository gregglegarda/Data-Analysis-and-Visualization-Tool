
import pandas as pd
import pip
#pip.main(['install', 'git+https://github.com/geopy/geopy'])
#pip.main(['install', 'git+https://github.com/python-visualization/folium'])
from geopy.geocoders import Nominatim, GoogleV3
import folium
from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView,QWebEnginePage as QWebPage
from PyQt5.QtWebEngineWidgets import QWebEngineSettings as QWebSettings
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QGroupBox, QGridLayout
from PyQt5.QtCore import QUrl
import numpy as np
import sys
import os

#########-------------------------------------- DATA FRAME -------------------------------------- #########
#data = pd.read_csv("https://www.kaggle.com/sobhanmoosavi/us-accidents#US_Accidents_Dec19.csv")
#data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_february_us_airport_traffic.csv')
print("--------------------------BUILD DATA FRAME--------------------------")
print("Building Dataframe...")
dataframe = pd.read_csv("US_Accidents_Dec19.csv")
# take random sample number of the 2 million rows
data = dataframe.sample(500)
print("Dataframe Created.")






#########-------------------------------------- DATA CLEANUP -------------------------------------- #########
#delete uneccessary columns
del data['ID']
del data['Source']
del data['TMC']
del data['End_Lat']
del data['End_Lng']
del data['Country']

#change datatypes and fill in nans
data['Description'] = data['Description'].fillna(0)
data['Street'] = data['Street'].fillna(0)
data['City'] = data['City'].fillna(0)
data['State'] = data['State'].fillna(0)
data['Zipcode'] = data['Zipcode'].fillna(0)
data['Description'] = np.where(pd.isnull(data['Description']),data['Description'],data['Description'].astype(str))
data['Street'] = np.where(pd.isnull(data['Street']),data['Street'],data['Street'].astype(str))
data['City'] = np.where(pd.isnull(data['City']),data['City'],data['City'].astype(str))
data['State'] = np.where(pd.isnull(data['State']),data['State'],data['State'].astype(str))
data['Zipcode'] = np.where(pd.isnull(data['Zipcode']),data['Zipcode'],data['Zipcode'].astype(str))


#print results
print("--------------------------COLUMNS--------------------------\n", data.columns)
print("--------------------------DATA TYPES--------------------------:\n", data.dtypes)
print("--------------------------HEAD--------------------------\n", data.head(10))
print("--------------------------NUMBER OF SAMPLES--------------------------\n",
      len(data),"Data points randomly selected from ", len(dataframe), "\n\n")


#########-------------------------------------- CREATE APPLICATION  -------------------------------------- #########
print("--------------------------APPLICATION--------------------------\nRunning Application...")
#create application instance. Should only be one running at a time
app = QApplication(sys.argv)


#########-------------------------------------- CREATE MAP -------------------------------------- #########
#create Qwebview Map instance
from Code import map_view
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "map.html"))
mapinstance = map_view.map_webview(file_path, data) #pass datapoints



#########-------------------------------------- CREATE WINDOW -------------------------------------- #########
#create window instance and put the map in
from Code import main_window
mainrun = main_window.runit(app, mapinstance)




#browser = QWebView()
#file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "map.html"))
#local_url = QUrl.fromLocalFile(file_path)
#browser.load(local_url)
#browser.show()






#app.exec_()