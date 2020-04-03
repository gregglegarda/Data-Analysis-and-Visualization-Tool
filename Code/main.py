
import pandas as pd

from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView,QWebEnginePage as QWebPage
from PyQt5.QtWebEngineWidgets import QWebEngineSettings as QWebSettings
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QGroupBox, QGridLayout
from PyQt5.QtCore import QUrl
import numpy as np
import sys
import os
#########-------------------------------------- AUTO IMPORTS  -------------------------------------- #########
import pip
#pip.main(['install', 'git+https://github.com/geopy/geopy'])
#pip.main(['install', 'git+https://github.com/python-visualization/folium'])
from geopy.geocoders import Nominatim, GoogleV3
import folium


#########-------------------------------------- PRE PROCESSING  -------------------------------------- #########
print("--------------------------PRE PROCESSING--------------------------")

datafile = "US_Accidents_Dec19.csv"
sample_size = 500

from Code import pre_process
data_instance = pre_process.data_frame(datafile, sample_size)
data = data_instance.create_dataframe()
data_instance.cleanup_data()


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



