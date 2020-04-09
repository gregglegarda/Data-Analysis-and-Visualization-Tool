
from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView,QWebEnginePage as QWebPage
from PyQt5.QtWebEngineWidgets import QWebEngineSettings as QWebSettings
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QGroupBox, QGridLayout
from PyQt5.QtCore import QUrl
import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QTableView, QTabWidget, QWidget, QVBoxLayout,
                            QGridLayout, QGroupBox,
                            QLabel, QLineEdit, QLCDNumber, QPushButton, QFrame,
                            QMessageBox, QAbstractItemView,
                            QPlainTextEdit)
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import csv
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, QObject
import sys
import os
import time
import ntpath
import qdarkstyle

def runit(app):

    gui = main_window(app)
    run = app.exec_()
    return gui, run

class main_window(QMainWindow):
    def __init__(self,app):
        self.app = app
        self.points=0



        super(main_window, self).__init__()
        #==================# MAIN WIDGET LAYOUT #==================#

        #Big window
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QGridLayout())
        self.widget.layout().setContentsMargins(50, 10, 50, 20)
        self.widget.layout().setSpacing(10)
        self.widget.layout().setColumnStretch(0, 4)
        self.widget.layout().setColumnStretch(1, 5)
        self.widget.layout().setColumnStretch(2, 7)
        self.widget.layout().setRowStretch(0, 6)
        self.widget.layout().setRowStretch(1, 7)
        self.widget.layout().setRowStretch(2, 13)
        self.widget.layout().setRowStretch(3, 15)
        self.setWindowTitle("US Accidents Data Mining")
        self.showMaximized()
        #THEME COLOR
        self.setStyleSheet("QMainWindow {background-image: url(background.jpg)}")
        print("Main window opened")

        # ==================# GROUP WIDGET LAYOUT #==================#
        # Small group0 (training)
        self.GroupBox0 = QGroupBox()
        self.layout0 = QGridLayout()
        self.GroupBox0.setLayout(self.layout0)
        self.layout0.setContentsMargins(20, 5, 20, 5)
        self.layout0.setSpacing(5)
        self.widget.layout().addWidget(self.GroupBox0, 0, 0, 2, 1)

        # Small group1 (predicting)
        self.GroupBox1 = QGroupBox()
        layout1 = QGridLayout()
        self.GroupBox1.setLayout(layout1)
        layout1.setContentsMargins(20, 5, 20, 5)
        layout1.setSpacing(5)
        self.widget.layout().addWidget(self.GroupBox1, 2, 0,3 , 1)

        #Small group (status)
        self.GroupBox_status = QGroupBox()
        self.layoutstatus = QGridLayout()
        self.GroupBox_status.setLayout(self.layoutstatus)
        self.layoutstatus.setContentsMargins(5, 5, 5, 5)
        self.layoutstatus.setSpacing(5)
        self.widget.layout().addWidget(self.GroupBox_status, 0, 1, 1, 1)


        # Small group2 (tabs)
        self.GroupBox2 = QGroupBox()
        layout2 = QGridLayout()
        self.GroupBox2.setLayout(layout2)
        layout2.setContentsMargins(5, 5, 5, 5)
        layout2.setSpacing(5)
        self.widget.layout().addWidget(self.GroupBox2, 1, 1, 2, 1)

        # Small group3 (summary of stats)
        self.GroupBox3 = QGroupBox()
        layout3 = QGridLayout()
        self.GroupBox3.setLayout(layout3)
        layout3.setContentsMargins(5, 5, 5, 5)
        layout3.setSpacing(5)
        self.widget.layout().addWidget(self.GroupBox3, 3, 1, 2, 2)

        # Small group4 (map)
        self.GroupBox4 = QGroupBox()
        self.layout4 = QGridLayout()
        self.GroupBox4.setLayout(self.layout4)
        self.layout4.setContentsMargins(5, 5, 5, 5)
        self.layout4.setSpacing(5)
        self.widget.layout().addWidget(self.GroupBox4, 0, 2, 3, 1)

        # ==================# GROUP 0 WIDGETS #==================#

        # Number of Samples
        self.num_samples = QLabel("Number of Samples:")
        self.layout0.addWidget(self.num_samples, 0, 0, 1, 1)
        self.SpinBox1 = QtWidgets.QSpinBox(self.widget)
        self.SpinBox1.setMaximum(500000)
        self.SpinBox1.setMinimum(100)
        self.SpinBox1.setSingleStep(100)
        self.layout0.addWidget(self.SpinBox1, 0, 1, 1, 1)

        # Train Percent
        self.train_percent = QLabel("Train Split:")
        self.layout0.addWidget(self.train_percent, 1, 0, 1, 1)
        self.SpinBox2 = QtWidgets.QSpinBox(self.widget)
        self.SpinBox2.setMaximum(90)
        self.SpinBox2.setMinimum(70)
        self.SpinBox2.setSingleStep(10)
        self.SpinBox2.setSuffix("%")
        self.layout0.addWidget(self.SpinBox2, 1, 1, 1, 1)

       

        # Test Percent
        self.model_algorithm_label = QLabel("Algorithm:")
        self.layout0.addWidget(self.model_algorithm_label, 2, 0, 1, 1)
        self.model_algorithm_combo = QtWidgets.QComboBox(self.widget)
        self.model_algorithm_combo.addItem("Decision Trees")
        self.model_algorithm_combo.addItem("Random Forest")
        self.model_algorithm_combo.addItem("Regression")
        self.layout0.addWidget(self.model_algorithm_combo, 2, 1, 1, 1)

        # Model Accuracy
        self.accuracy = QLabel("Model Accuracy:")
        self.layout0.addWidget(self.accuracy, 3, 0, 1, 1)
        self.accuracy_display = QLCDNumber()
        self.accuracy_display.setMaximumHeight(50)
        self.accuracy_display.setSegmentStyle(QLCDNumber.Flat)
        self.layout0.addWidget(self.accuracy_display, 3, 1, 1, 1)


        # Train Model
        Button1 = QtWidgets.QPushButton(self.widget)
        Button1.setText("Train Model")
        Button1.clicked.connect(self.on_Button_train_clicked)
        self.layout0.addWidget(Button1, 4, 0, 1, 2)



        # ==================# GROUP 1 WIDGETS #==================#


        #Latitude
        self.latitude = QLabel("Latitude:")
        layout1.addWidget(self.latitude, 0, 0, 1, 1)
        self.line_edit_latitude = QLineEdit()
        self.line_edit_latitude.setPlaceholderText('Enter Latitude')
        layout1.addWidget(self.line_edit_latitude, 0, 1, 1, 1)

        #Longitude
        self.longitude = QLabel("Longitude:")
        layout1.addWidget(self.longitude, 1, 0, 1, 1)
        self.line_edit_longitude = QLineEdit()
        self.line_edit_longitude.setPlaceholderText('Enter Longitude')
        layout1.addWidget(self.line_edit_longitude, 1, 1, 1, 1)

        #Distance(mi)
        self.distance = QLabel("Distance:")
        layout1.addWidget(self.distance, 2, 0, 1, 1)
        self.line_edit_distance = QLineEdit()
        self.line_edit_distance.setPlaceholderText('Enter Distance')
        layout1.addWidget(self.line_edit_distance, 2, 1, 1, 1)

        #Number
        self.number = QLabel("Number:")
        layout1.addWidget(self.number, 3, 0, 1, 1)
        self.line_edit_number = QLineEdit()
        self.line_edit_number.setPlaceholderText('Enter Number')
        layout1.addWidget(self.line_edit_number, 3, 1, 1, 1)

        #Temperature
        self.temperature = QLabel("Temperature:")
        layout1.addWidget(self.temperature, 4, 0, 1, 1)
        self.line_edit_temperature = QLineEdit()
        self.line_edit_temperature.setPlaceholderText('Enter Temperature')
        layout1.addWidget(self.line_edit_temperature, 4, 1, 1, 1)

        #Wind Chill(F)
        self.wind_chill = QLabel("Wind Chill:")
        layout1.addWidget(self.wind_chill, 5, 0, 1, 1)
        self.line_edit_wind_chill = QLineEdit()
        self.line_edit_wind_chill.setPlaceholderText('Enter Wind Chill')
        layout1.addWidget(self.line_edit_wind_chill, 5, 1, 1, 1)

        #Humidity%
        self.humidity = QLabel("Humidity:")
        layout1.addWidget(self.humidity, 6, 0, 1, 1)
        self.line_edit_humidity = QLineEdit()
        self.line_edit_humidity.setPlaceholderText('Enter Humidity')
        layout1.addWidget(self.line_edit_humidity, 6, 1, 1, 1)

        #Pressure(in)
        self.pressure = QLabel("Pressure:")
        layout1.addWidget(self.pressure, 7, 0, 1, 1)
        self.line_edit_pressure = QLineEdit()
        self.line_edit_pressure.setPlaceholderText('Enter Pressure')
        layout1.addWidget(self.line_edit_pressure, 7, 1, 1, 1)

        #Visibility(mi)
        self.visibility = QLabel("Visibility:")
        layout1.addWidget(self.visibility, 8, 0, 1, 1)
        self.line_edit_visibility = QLineEdit()
        self.line_edit_visibility.setPlaceholderText('Enter Visibility')
        layout1.addWidget(self.line_edit_visibility, 8, 1, 1, 1)

        #Wind Speed(mph)
        self.wind_speed = QLabel("Wind Speed:")
        layout1.addWidget(self.wind_speed, 9, 0, 1, 1)
        self.line_edit_wind_speed = QLineEdit()
        self.line_edit_wind_speed.setPlaceholderText('Enter Wind Speed')
        layout1.addWidget(self.line_edit_wind_speed, 9, 1, 1, 1)

        #Precipitation(in)
        self.precipitation = QLabel("Precipitation:")
        layout1.addWidget(self.precipitation, 10, 0, 1, 1)
        self.line_edit_precipitation = QLineEdit()
        self.line_edit_precipitation.setPlaceholderText('Enter Precipitation')
        layout1.addWidget(self.line_edit_precipitation, 10, 1, 1, 1)

        # Severity
        self.severity = QLabel("Severity Prediction:")
        layout1.addWidget(self.severity, 11, 0, 1, 1)
        self.severity_display = QLCDNumber()
        self.severity_display.setMaximumHeight(50)
        self.severity_display.setSegmentStyle(QLCDNumber.Flat)
        layout1.addWidget(self.severity_display, 11 , 1, 1, 1)


        # predict button
        Button_predict = QtWidgets.QPushButton(self.widget)
        Button_predict.setText("Predict Severity of Accident")
        Button_predict.clicked.connect(self.on_Button_predict_clicked)
        layout1.addWidget(Button_predict, 12, 0, 1, 2)

        # ==================# GROUP STATUS WIDGETS (STATUS LAYOUT) #==================#


        self.status_label_model = QLabel("ML MODEL:")
        self.layoutstatus.addWidget(self.status_label_model, 0, 0, 1, 1)
        self.status_display_model = QLCDNumber()
        self.status_display_model.display("oooo")
        self.status_display_model.setStyleSheet("QLCDNumber { color: black ; background-color: yellow}")
        self.status_display_model.setMaximumHeight(25)
        self.status_display_model.setSegmentStyle(QLCDNumber.Flat)
        self.layoutstatus.addWidget(self.status_display_model, 1, 0, 1, 1)

        self.status_map = QLabel("GEOMAP:")
        self.layoutstatus.addWidget(self.status_map, 0, 1, 1, 1)
        self.status_display_map = QLCDNumber()
        self.status_display_map.display("ready")
        self.status_display_map.setStyleSheet("QLCDNumber { color: black ; background-color: green}")
        self.status_display_map.setMaximumHeight(25)
        self.status_display_map.setSegmentStyle(QLCDNumber.Flat)
        self.layoutstatus.addWidget(self.status_display_map, 1, 1, 1, 1)






        # ==================# GROUP 2 WIDGETS (TABS WIDGET LAYOUT) #==================#
        #Tabs
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        #self.tabs.resize(300, 200)


        # Add tabs
        #tab1
        self.tabs.addTab(self.tab1, "Histograms")
        tab1_layout = QVBoxLayout(self)
        self.tab1.setLayout(tab1_layout)

        # tab2
        self.tabs.addTab(self.tab2, "HeatMap")
        self.tab2_layout = QVBoxLayout(self)
        self.tab2.setLayout(self.tab2_layout)

        # tab3
        self.tabs.addTab(self.tab3, "Correlogram")
        tab3_layout = QVBoxLayout(self)
        self.tab3.setLayout(tab3_layout)

        # tab4
        self.tabs.addTab(self.tab4, "Boxplots")
        tab4_layout = QVBoxLayout(self)
        self.tab4.setLayout(tab4_layout)

        # Add tabs to widget
        layout2.addWidget(self.tabs)

        # ==================# INDIVIDUAL TAB WIDGETS (INSIDE GROUP 2)#==================#

        # HISTOGRAMS
        self.imageView = QLabel(self.widget)
        self.pixmap = QPixmap("analysis.png")
        self.imageView.setPixmap(self.pixmap)
        # scroller
        #self.scroll = QtWidgets.QScrollArea(self.widget)
        #self.scroll.setWidget(self.imageView)
        tab1_layout.addWidget(self.imageView)

        # CORRELATION
        self.imageView3 = QLabel(self.widget)
        self.pixmap3 = QPixmap("analysis.png")
        self.imageView3.setPixmap(self.pixmap3)
        # scroller
        # self.scroll = QtWidgets.QScrollArea(self.widget)
        # self.scroll.setWidget(self.imageView)
        tab3_layout.addWidget(self.imageView3)



        # ==================# GROUP 3 WIDGETS (TABLE DATABASE) #==================#
        filename = os.path.expanduser(os.path.abspath(os.path.join(os.path.dirname(__file__), "statistic_summary.csv")))
        self.items = []
        self.fileName = filename
        #self.on_Button1_clicked

        self.model = QtGui.QStandardItemModel(self.widget)

        self.model.setHorizontalHeaderLabels(
            ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
        self.model.setVerticalHeaderLabels(
            ['Severity', 'Latitude', 'Longitude', 'Distance(mi)', 'Number', 'Temperature',
             'Wind Chill(F)', 'Humidity%', 'Pressure(in)', 'Visibility(mi)',
             'Wind Speed(mph)', 'Precipitation(in)'])
        self.tableView = QTableView(self.widget)
        #self.tableView.setStyleSheet("QTableView{ background-color: rgb(45, 45, 45);  }")  # cell color
        self.tableView.horizontalHeader().setStretchLastSection(True)
        self.tableView.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.tableView.setModel(self.model)
        layout3.addWidget(self.tableView, 1, 0, 1, 4)


        # ==================# GROUP 4 WIDGETS (MAP) #==================#
        # Qwebview maps
        #create map instance and update later when the train button is clicked
        self.map = 0
        self.create_map_instance()
        self.layout4.addWidget(self.map)



        # ==================# SHOW EVERYTHING #==================#
        self.show()

#########################################   FUNCTIONS   #####################################################

    #=============== VIEW STATISTICAL SUMMRY FUNCTION ====================#
   # @QtCore.pyqtSlot()
    #def on_Button1_clicked(self):
        #self.loadCsv(self.fileName)
        #print("Statistical summary button clicked")

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
                ['Severity', 'Latitude', 'Longitude', 'Distance(mi)', 'Number', 'Temperature',
                 'Wind Chill(F)', 'Humidity%', 'Pressure(in)', 'Visibility(mi)',
                 'Wind Speed(mph)', 'Precipitation(in)'])
        except:
            self.model.setVerticalHeaderLabels(
                ['Severity', 'Latitude', 'Longitude', 'Distance(mi)', 'Number', 'Temperature',
                 'Wind Chill(F)', 'Humidity%', 'Pressure(in)', 'Visibility(mi)',
                 'Wind Speed(mph)', 'Precipitation(in)'])
            print("No Database")



    #====================== GET ATTRIBUTES  FUNCTION=========================#

    def get_predict_attributes(self):
        print(
            "Data in predict button is: ",
            #self.severity_display.text(),
            self.line_edit_latitude.text(),
            self.line_edit_longitude.text(),
            self.line_edit_distance.text(),
            self.line_edit_number.text(),
            self.line_edit_temperature.text(),
            self.line_edit_wind_chill.text(),
            self.line_edit_humidity.text(),
            self.line_edit_pressure.text(),
            self.line_edit_visibility.text(),
            self.line_edit_wind_speed.text(),
            self.line_edit_precipitation.text()
        )
        return(
            # self.severity_display.text(),
            self.line_edit_latitude.text(),
            self.line_edit_longitude.text(),
            self.line_edit_distance.text(),
            self.line_edit_number.text(),
            self.line_edit_temperature.text(),
            self.line_edit_wind_chill.text(),
            self.line_edit_humidity.text(),
            self.line_edit_pressure.text(),
            self.line_edit_visibility.text(),
            self.line_edit_wind_speed.text(),
            self.line_edit_precipitation.text()
        )

    def get_train_attributes(self):
        print(
            "Data in train button is: ",
            self.SpinBox1.text(),
            self.SpinBox2.text(),
            self.model_algorithm_combo.currentText(),
            #self.accuracy_display.text()
        )
        return(
            self.SpinBox1.text(),
            self.SpinBox2.text(),
            self.model_algorithm_combo.currentText(),
            # self.accuracy_display.text()
        )

    #########-------------------------------------- TRAIN FUNCTION -------------------------------------- #########
    def on_Button_train_clicked(self):
        print("Button train clicked")

        self.update_status_bar()
        self.app.processEvents()
        attributes = self.get_train_attributes()
        try:
            import train_model
        except:
            print("import exception")
        model1 = train_model.train(attributes)
        self.points = model1.get_map_data_points()
        self.accuracy_display.display(model1.get_model_accuracy())  # set the lcd accuract digit
        self.update_screen_widgets()

        #self.app.processEvents()
        #time.sleep(10)
        self.status_display_map.display("ready")
        self.status_display_map.setStyleSheet("QLCDNumber { color: black ; background-color: green}")
        self.status_display_map.update()

    def update_status_bar(self):
        self.status_display_model.display("busy")
        self.status_display_model.setStyleSheet("QLCDNumber { color: black ; background-color: red}")
        self.status_display_model.update()
        self.status_display_map.display("busy")
        self.status_display_map.setStyleSheet("QLCDNumber { color: black ; background-color: red}")
        self.status_display_map.update()



    def update_screen_widgets(self):
        # update image in screen
        self.pixmap = QPixmap("temp_hist.png")
        self.imageView.setPixmap(self.pixmap)
        self.imageView.update()

        # update image in screen
        self.pixmap3 = QPixmap("correlation_matrix.png")
        self.imageView3.setPixmap(self.pixmap3)
        self.imageView3.update()

        # update map
        self.layout4.removeWidget(self.map)
        self.create_map_instance()
        # self.map.update()
        self.layout4.addWidget(self.map)

        # update summary table
        filename = os.path.expanduser(os.path.abspath(os.path.join(os.path.dirname(__file__), "statistic_summary.csv")))
        self.items = []
        self.fileName = filename
        self.loadCsv(self.fileName)

        #update lcd train
        #self.accuracy_display.setStyleSheet("QLCDNumber { color: black ; background-color: #1f77b4 }")
        self.accuracy_display.update()


        #status bar update
        self.status_display_model.display("ready")
        self.status_display_model.setStyleSheet("QLCDNumber { color: black ; background-color: green}")
        self.status_display_model.update()
        self.status_display_map.display("busy")
        self.status_display_map.setStyleSheet("QLCDNumber { color: black ; background-color: yellow}")
        self.status_display_map.update()






    #########-------------------------------------- PREDICT FUNCTION -------------------------------------- #########
    def on_Button_predict_clicked(self):

        print("Button predict clicked")
        attributes = self.get_predict_attributes()
        try:
            import predict
        except:
            print("import exception")
        predict1 = predict.predict(attributes)

    #########-------------------------------------- INITIALIZE MAP FUNCTION -------------------------------------- #########
    def create_map_instance(self):

        # create Qwebview Map instance
        try:
            import map_view
        except:
            print("import exception")

        # create initial dummy data for map
        #initial self.points is 0
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "map.html"))
        self.map = map_view.map_webview(file_path, self.points)  # pass datapoints
