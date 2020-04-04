
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
        self.widget.layout().setContentsMargins(20, 5, 20, 20)
        self.widget.layout().setSpacing(5)
        self.widget.layout().setColumnStretch(0, 3)
        self.widget.layout().setColumnStretch(1,1)
        self.widget.layout().setColumnStretch(2, 2)
        self.widget.layout().setRowStretch(0, 1)
        self.widget.layout().setRowStretch(1, 4)
        self.setWindowTitle("US Accidents Data Mining")
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
        self.widget.layout().addWidget(self.GroupBox1, 0, 0, 1, 2)

        # Small group2
        self.GroupBox2 = QGroupBox()
        layout2 = QGridLayout()
        self.GroupBox2.setLayout(layout2)
        layout2.setContentsMargins(5, 5, 5, 5)
        layout2.setSpacing(5)
        self.widget.layout().addWidget(self.GroupBox2, 0, 2, 2, 1)

        # Small group3
        self.GroupBox3 = QGroupBox()
        layout3 = QGridLayout()
        self.GroupBox3.setLayout(layout3)
        layout3.setContentsMargins(5, 5, 5, 5)
        layout3.setSpacing(5)
        self.widget.layout().addWidget(self.GroupBox3, 2, 0, 1, 3)

        # ==================# TABS WIDGET LAYOUT #==================#
        #Tabs
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        self.tab5 = QWidget()
        #self.tabs.resize(300, 200)


        # Add tabs
        #tab1
        self.tabs.addTab(self.tab1, "Histograms")
        tab1_layout = QVBoxLayout(self)
        self.tab1.setLayout(tab1_layout)

        # tab2
        self.tabs.addTab(self.tab2, "GeoMap")
        tab2_layout = QVBoxLayout(self)
        self.tab2.setLayout(tab2_layout)

        # tab3
        self.tabs.addTab(self.tab3, "Correlation")
        tab3_layout = QVBoxLayout(self)
        self.tab3.setLayout(tab3_layout)

        # tab4
        self.tabs.addTab(self.tab4, "Boxplots")
        tab4_layout = QVBoxLayout(self)
        self.tab4.setLayout(tab4_layout)

        # tab5
        self.tabs.addTab(self.tab5, "Summary")
        tab5_layout = QVBoxLayout(self)
        self.tab5.setLayout(tab5_layout)

        # Add tabs to widget
        layout2.addWidget(self.tabs)




        # ==================# ADDED WIDGETS #==================#
        #Qwebview maps
        tab2_layout.addWidget(self.map)

        #Button 1
        Button1 = QtWidgets.QPushButton(self.widget)
        Button1.setText("Preprocess Random Sample")
        Button1.clicked.connect(self.on_Button1_clicked)
        self.widget.layout().addWidget(Button1, 1, 0, 1, 1)

        # spinbox1
        SpinBox1 = QtWidgets.QSpinBox(self.widget)
        SpinBox1.setMaximum(1000000)
        self.widget.layout().addWidget(SpinBox1, 1, 1, 1, 1)

        #button2
        Button2 = QtWidgets.QPushButton(self.widget)
        Button2.setText("Button2")
        Button2.clicked.connect(self.on_Button2_clicked)
        layout1.addWidget(Button2, 0, 1, 1, 1)




        # Image
        # image box
        self.imageView = QLabel(self.widget)
        self.pixmap = QPixmap("temp_hist.png")
        self.imageView.setPixmap(self.pixmap)
        # scroller
        #self.scroll = QtWidgets.QScrollArea(self.widget)
        #self.scroll.setWidget(self.imageView)
        tab1_layout.addWidget(self.imageView)

        # ==================# TABLE DATABASE #==================#
        filename = os.path.expanduser(os.path.abspath(os.path.join(os.path.dirname(__file__), "statistic_summary.csv")))
        self.items = []
        self.fileName = filename
        self.on_Button1_clicked

        self.model = QtGui.QStandardItemModel(self.widget)

        self.model.setHorizontalHeaderLabels(
            ['Severity', 'Latitude', 'Longitude', 'Distance(mi)', 'Number', 'Temperature',
             'Wind Chill(F)', 'Humidity%', 'Pressure(in)', 'Visibility(mi)',
             'Wind Speed(mph)', 'Precipitation(in)'])
        self.model.setVerticalHeaderLabels(
            ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
        self.tableView = QTableView(self.widget)
        #self.tableView.setStyleSheet("QTableView{ background-color: rgb(45, 45, 45);  }")  # cell color
        self.tableView.horizontalHeader().setStretchLastSection(True)
        self.tableView.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.tableView.setModel(self.model)
        layout3.addWidget(self.tableView, 1, 0, 1, 4)

        # ==================# SHOW EVERYTHING #==================#
        self.show()

#########################################   FUNCTIONS   #####################################################
    @QtCore.pyqtSlot()
    def on_Button1_clicked(self):
        self.loadCsv(self.fileName)
        print("button1 clicked")

    def loadCsv(self, fileName):
        while (self.model.rowCount() > 0):
            self.model.removeRow(0)
        try:
            with open(fileName, "r") as fileInput:
                for row in csv.reader(fileInput):
                    self.items = [
                        QtGui.QStandardItem(field)
                        for field in row
                    ]
                    self.model.appendRow(self.items)
            self.model.setVerticalHeaderLabels(
                ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
        except:
            print("No Database")

    def on_Button2_clicked(self):
        print("button 2 clicked")

    def eda_tab(self):
        print("eda tab clicked")