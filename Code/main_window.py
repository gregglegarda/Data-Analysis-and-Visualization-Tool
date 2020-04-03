
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
from PyQt5.QtWidgets import (QMainWindow, QTableView,
                            QGridLayout, QGroupBox,
                            QLabel, QLineEdit,  QPushButton,
                            QMessageBox, QAbstractItemView,
                            QPlainTextEdit)
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import csv
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QItemSelectionModel
import sys
import os
import ntpath

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
        print("Opened Main Window")

        #Small group1
        self.GroupBox1 = QGroupBox()
        layout1 = QGridLayout()
        self.GroupBox1.setLayout(layout1)
        layout1.setContentsMargins(5, 5, 5, 5)
        layout1.setSpacing(5)
        self.widget.layout().addWidget(self.GroupBox1, 0, 0,1,1)

        # Small group2
        self.GroupBox2 = QGroupBox()
        layout2 = QGridLayout()
        self.GroupBox2.setLayout(layout2)
        layout2.setContentsMargins(5, 5, 5, 5)
        layout2.setSpacing(5)
        self.widget.layout().addWidget(self.GroupBox2, 0, 1, 1, 3)

        # ==================# ADDED WIDGETS #==================#
        #Qwebview map
        layout1.addWidget(self.map, 0, 0, 1, 3)

        #Buttons
        Button1 = QtWidgets.QPushButton(self.widget)
        Button1.setText("Button1")
        Button1.clicked.connect(self.on_Button1_clicked)
        layout2.addWidget(Button1, 0, 0, 1, 1)

        # TRUE ABNORMAL BUTTON
        Button2 = QtWidgets.QPushButton(self.widget)
        Button2.setText("Button2")
        Button2.clicked.connect(self.on_Button2_clicked)
        layout2.addWidget(Button2, 0, 1, 1, 1)


        self.show()



    def on_Button1_clicked(self):
        print("button 1 clicked")

    def on_Button2_clicked(self):
        print("button 2 clicked")
