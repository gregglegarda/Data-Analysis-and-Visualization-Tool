
from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView,QWebEnginePage as QWebPage
from PyQt5.QtWebEngineWidgets import QWebEngineSettings as QWebSettings
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QGroupBox, QGridLayout
from PyQt5.QtCore import QUrl
import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QTableView, QTabWidget, QWidget, QVBoxLayout,
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
import qdarkstyle

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
        self.widget.layout().setColumnStretch(0, 3)
        self.widget.layout().setColumnStretch(1,1)
        self.widget.layout().setRowStretch(0, 3)
        self.widget.layout().setRowStretch(1, 2)
        self.setWindowTitle("Main Window")
        self.showMaximized()
        #THEME COLOR
        self.setStyleSheet("QMainWindow {background-image: url(background.jpg)}")
        print("Opened Main Window")

        # ==================# GROUP WIDGET LAYOUT #==================#
        # Small group1
        self.GroupBox1 = QGroupBox()
        layout1 = QGridLayout()
        self.GroupBox1.setLayout(layout1)
        layout1.setContentsMargins(5, 5, 5, 5)
        layout1.setSpacing(5)
        self.widget.layout().addWidget(self.GroupBox1, 0, 0, 1, 1)

        # Small group2
        self.GroupBox2 = QGroupBox()
        layout2 = QGridLayout()
        self.GroupBox2.setLayout(layout2)
        layout2.setContentsMargins(5, 5, 5, 5)
        layout2.setSpacing(5)
        self.widget.layout().addWidget(self.GroupBox2, 1, 0, 1, 1)

        # Small group3
        self.GroupBox3 = QGroupBox()
        layout3 = QGridLayout()
        self.GroupBox3.setLayout(layout3)
        layout3.setContentsMargins(5, 5, 5, 5)
        layout3.setSpacing(5)
        self.widget.layout().addWidget(self.GroupBox3, 0, 1, 2, 1)

        # ==================# TABS WIDGET LAYOUT #==================#
        #Tabs
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tabs.resize(300, 200)


        # Add tabs
        #tab1
        self.tabs.addTab(self.tab1, "Temperature")
        tab1_layout = QVBoxLayout(self)
        self.tab1.setLayout(tab1_layout)

        # tab2
        self.tabs.addTab(self.tab2, "Tab 2")


        # Add tabs to widget
        layout3.addWidget(self.tabs)




        # ==================# ADDED WIDGETS #==================#
        #Qwebview maps
        layout1.addWidget(self.map, 0, 0, 1, 3)

        #Button 1
        Button1 = QtWidgets.QPushButton(self.widget)
        Button1.setText("Button 1")
        Button1.clicked.connect(self.on_Button1_clicked)
        layout2.addWidget(Button1, 0, 0, 1, 1)

        #button2
        Button2 = QtWidgets.QPushButton(self.widget)
        Button2.setText("Button2")
        Button2.clicked.connect(self.on_Button2_clicked)
        layout2.addWidget(Button2, 0, 1, 1, 1)


        # Image
        # image box
        self.imageView = QLabel(self.widget)
        self.pixmap = QPixmap("temp_hist.png")
        self.imageView.setPixmap(self.pixmap)
        # scroller
        #self.scroll = QtWidgets.QScrollArea(self.widget)
        #self.scroll.setWidget(self.imageView)
        tab1_layout.addWidget(self.imageView)





        self.show()



    def on_Button1_clicked(self):
        print("button 1 clicked")

    def on_Button2_clicked(self):
        print("button 2 clicked")

    def eda_tab(self):
        print("eda tab clicked")