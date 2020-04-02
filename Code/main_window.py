
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

def runit(app, map):

    gui = main_window(app, map)
    run = app.exec_()
    return gui, run

class main_window(QMainWindow):
    def __init__(self,app, map):
        self.app = app
        self.map = map
        super(main_window, self).__init__()
        #==================# MAIN WIDGET LAYOUT #==================#

        #Big window
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QGridLayout())
        self.widget.layout().setContentsMargins(20, 20, 20, 20)
        self.widget.layout().setSpacing(5)
        self.setWindowTitle("Main Window")
        self.showMaximized()
        #THEME COLOR
        self.setStyleSheet("QMainWindow {background-image: url(background/background.jpg)}")
        print("Doctor GUI Screen")

        #Small group1
        self.GroupBox1 = QGroupBox()
        layout1 = QGridLayout()
        self.GroupBox1.setLayout(layout1)
        layout1.setContentsMargins(5, 5, 5, 5)
        layout1.setSpacing(5)
        self.widget.layout().addWidget(self.GroupBox1, 0, 0,1,3)

        #Qwebview map
        layout1.addWidget(self.map, 0, 0, 1, 3)



        self.show()



