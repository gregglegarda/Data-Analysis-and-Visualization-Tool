
import pandas as pd

from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView,QWebEnginePage as QWebPage
from PyQt5.QtWebEngineWidgets import QWebEngineSettings as QWebSettings
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QGroupBox, QGridLayout
from PyQt5.QtCore import QUrl
import numpy as np
from PyQt5 import Qt
help(Qt)
from PyQt5.Qt import QPalette, QColor
import sys
import os
import pip
#########-------------------------------------- AUTO IMPORTS  -------------------------------------- #########

try:
    from geopy.geocoders import Nominatim, GoogleV3
    import folium
    import qdarkstyle
except:
    print("Importing packages...")
    #pip.main(['install', 'git+https://github.com/geopy/geopy'])
    #pip.main(['install', 'git+https://github.com/python-visualization/folium'])
    os.system('pip install geopy')
    os.system('pip install folium')
    os.system('pip install qdarkstyle')
    print("Imported packages")



#########-------------------------------------- PATHS  -------------------------------------- #########
print("--------------------------PATHS--------------------------")
print("Creating path...")
try:
    from Code import pre_process
    from Code import map_view
    from Code import main_window
    from Code import eda_stats

except:   #Following code (try/except clauses) searches for this script, and then changes the current working directory to the folder that houses it.

    try:
        start = '/Users'  #Code from https://stackoverflow.com/questions/43553742/finding-particular-path-in-directory-in-python
        for dirpath, dirnames, filenames in os.walk(start):
            for filename in filenames:
                if filename == "US_Accidents_Dec19.csv":
                    filename = os.path.join(dirpath, filename)
                    os.chdir(dirpath)

    except:
        start1 = "C:\\Users"
        for dirpath, dirnames, filenames in os.walk(start1):
            for filename in filenames:
                if filename == "US_Accidents_Dec19.csv":
                    filename = os.path.join(dirpath, filename)
                    os.chdir(dirpath)

print("Path Created")


#########-------------------------------------- DATA PROCESSING -------------------------------------- #########
print("--------------------------DATA PROCESSING--------------------------")

### try opening pre processed sample
try:
    pre_processed_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "pre_processed_data.csv"))
    data = pd.read_csv(pre_processed_path)
    print("Using preprocesses data")


### if not there, create a sample
except:
    datafile = "US_Accidents_Dec19.csv"
    sample_size = 10000
    try:
        import pre_process
    except:
        print("import exception")

    data_instance = pre_process.data_frame(datafile, sample_size)
    data = data_instance.create_dataframe()
    data_instance.cleanup_data()

#########-------------------------------------- DATA ANALYSIS -------------------------------------- #########
print("--------------------------DATA ANALYSIS--------------------------")
try:
    import eda_stats
except:
    print("import exception")
data_analysis = eda_stats.eda(data)
data_analysis.perform_eda()


#########-------------------------------------- CREATE APPLICATION  -------------------------------------- #########
print("--------------------------APPLICATION--------------------------\nRunning Application...")
#create application instance. Should only be one running at a time
app = QApplication(sys.argv)
app.setStyle("Fusion")
palette = QPalette()
palette.setColor(QPalette.Window, QColor(53, 53, 53))
palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
palette.setColor(QPalette.Base, QColor(25, 25, 25))
palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
palette.setColor(QPalette.Text, QColor(255, 255, 255))
palette.setColor(QPalette.Button, QColor(53, 53, 53))
palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
palette.setColor(QPalette.Link, QColor(42, 130, 218))
palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
app.setPalette(palette)


#########-------------------------------------- CREATE MAP -------------------------------------- #########
#create Qwebview Map instance
try:
    import map_view
except:
    print("import exception")

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "map.html"))
mapinstance = map_view.map_webview(file_path, data) #pass datapoints



#########-------------------------------------- CREATE WINDOW -------------------------------------- #########

#create window instance and put the map in
try:
    import main_window
except:
    print("import exception")

mainrun = main_window.runit(app, mapinstance)



