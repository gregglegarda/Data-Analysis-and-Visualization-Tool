
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
import sys
import os

#########-------------------------------------- DATA FRAME -------------------------------------- #########
#data = pd.read_csv("https://www.kaggle.com/sobhanmoosavi/us-accidents#US_Accidents_Dec19.csv")
#data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_february_us_airport_traffic.csv')

data = pd.read_csv("US_Accidents_Dec19.csv")
# take random sample number of the 2 million rows
data = data.sample(1000)
print(data.columns)




#########-------------------------------------- DATA CLEANUP -------------------------------------- #########

#delete uneccessary columns
del data['Source']
print(data.columns)
print(data.head(10))
print(len(data))


#########-------------------------------------- CREATE APPLICATION  -------------------------------------- #########
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