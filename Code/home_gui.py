#!/usr/bin/env python3
import csv
import numpy as np
import os
import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QFormLayout, QGroupBox, QApplication, QWidget, QPushButton, QLabel, QLineEdit, QGridLayout, QMessageBox,QInputDialog, QComboBox, QMainWindow)



def runit(app):
    gui = home(app)
    gui.show()
    run = app.exec_()
    return gui, run

def stop(run):
    sys.exit(run)

#create window class
class home(QMainWindow):
    def __init__(self,app):
        self.app = app
        super().__init__()
        self.successful= 0
        self.usertype = 0
        self.setWindowTitle('Hour Logger')
        self.resize(500, 190)


        ##the main widget layout
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QGridLayout())
        self.widget.layout().setContentsMargins(500, 250, 500, 250)
        self.widget.layout().setSpacing(1)
        self.widget.layout().setColumnMinimumWidth(0, 100)
        self.widget.layout().setColumnMinimumWidth(2, 100)
        self.widget.layout().setRowMinimumHeight(0, 100)
        self.widget.layout().setRowMinimumHeight(3, 100)
        self.showMaximized()

        button_createaccount = QPushButton('Create Account')
        button_createaccount.clicked.connect(self.button_createaccount_clicked)
        self.widget.layout().addWidget(button_createaccount, 1, 1)

        button_login = QPushButton('Login Here')
        button_login.clicked.connect(self.button_login_clicked)
        self.widget.layout().addWidget(button_login,2,1)

        self.setStyleSheet("QMainWindow {border-image: url(background/background.jpg) 0 0 0 0 stretch stretch;}")
        print("Home Screen")


    @QtCore.pyqtSlot()
    def button_login_clicked(self):
        import login
        loginapp = login.runit(self.app)
        self.successful, self.usertype = loginapp.successlogin()

    def login_button(self):
        return self.successful, self.usertype

    @QtCore.pyqtSlot()
    def button_createaccount_clicked(self):
        self.createaccount_button()
    def createaccount_button(self):
        x=0
        #from scripts import login
        #running = login.runit(self.app)

