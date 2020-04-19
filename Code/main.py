
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView,QWebEnginePage as QWebPage
from PyQt5.QtWebEngineWidgets import QWebEngineSettings as QWebSettings
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QGroupBox, QGridLayout
from PyQt5.QtCore import QUrl
import numpy as np
from PyQt5 import Qt
#help(Qt)
from PyQt5.Qt import QPalette, QColor
import sys
import os
import pip
import faulthandler
faulthandler.enable()
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



#########-------------------------------------- CREATE WINDOW -------------------------------------- #########

#create window instance and put the map in
try:
    import main_window
except:
    print("import exception")

mainrun = main_window.runit(app)
#mainrun.get_attributes











