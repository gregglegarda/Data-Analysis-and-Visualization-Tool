
import pandas as pd

from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView,QWebEnginePage as QWebPage
from PyQt5.QtWebEngineWidgets import QWebEngineSettings as QWebSettings
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QGroupBox, QGridLayout
from PyQt5.QtCore import QUrl
import numpy as np
import sys
import os
import pip
#########-------------------------------------- AUTO IMPORTS  -------------------------------------- #########

try:
    from geopy.geocoders import Nominatim, GoogleV3
    import folium
except:
    os.system('pip install geopy')
    os.system('pip install folium')
    pip.main(['install', 'git+https://github.com/geopy/geopy'])
    pip.main(['install', 'git+https://github.com/python-visualization/folium'])
    print("import exception occurred")



#########-------------------------------------- PRE PROCESSING  -------------------------------------- #########
print("--------------------------PRE PROCESSING--------------------------")

datafile = "US_Accidents_Dec19.csv"
sample_size = 5000

try:
    import pre_process

except:
    from Code import pre_process
    print("import exception")
data_instance = pre_process.data_frame(datafile, sample_size)
data = data_instance.create_dataframe()
data_instance.cleanup_data()


#########-------------------------------------- CREATE APPLICATION  -------------------------------------- #########
print("--------------------------APPLICATION--------------------------\nRunning Application...")
#create application instance. Should only be one running at a time
app = QApplication(sys.argv)


#########-------------------------------------- CREATE MAP -------------------------------------- #########
#create Qwebview Map instance
try:
    import map_view
except:
    from Code import main_window
    print("import exception")

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "map.html"))
mapinstance = map_view.map_webview(file_path, data) #pass datapoints



#########-------------------------------------- CREATE WINDOW -------------------------------------- #########

#create window instance and put the map in
try:
    import main_window
except:
    from Code import main_window
    print("import exception")

mainrun = main_window.runit(app, mapinstance)



