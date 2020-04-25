from PyQt5.QtWidgets import (QMainWindow, QApplication, QComboBox, QDialog,
                             QDialogButtonBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
                             QLabel, QLineEdit, QMenu, QMenuBar, QPushButton, QSpinBox, QTextEdit,
                             QVBoxLayout,QMessageBox)
from PyQt5.QtGui import QPalette,QColor, QBrush, QPixmap
from PyQt5 import Qt
import sys
import numpy as np
from datetime import datetime
from uuid import uuid4



class pop_up_entry(QDialog):
    NumGridRows = 3
    NumButtons = 4

    def __init__(self,app, model_name):
        self.app = app
        self.model_name = model_name
        self.k = 0
        self.submitted = False

        super(pop_up_entry, self).__init__()
        self.makeform()


        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBox)
        self.setLayout(mainLayout)

        self.setWindowTitle(model_name)
        #self.showMaximized()

        #THEME COLOR
        self.palette = self.palette()
        self.setPalette(self.palette)
        print("Pop up Entry Screen")
        self.exec()


    def makeform(self):
        self.formGroupBox = QGroupBox()
        layout = QFormLayout()
        self.line_edit_K = QLineEdit()
        self.line_edit_K.setPlaceholderText('Value')

        self.label_K = QLabel("K = ")

        button_submit = QPushButton('Submit')
        button_submit.clicked.connect(self.check_submit)


        layout.addRow(self.label_K, self.line_edit_K )
        layout.addRow(button_submit)

        self.formGroupBox.setLayout(layout)

    def check_submit(self):
        msg = QMessageBox()
        if (self.line_edit_K.text() != ''):
            self.k = self.line_edit_K.text()
            self.submitted = True
            msg.setText('Training Model')
            msg.exec_()
            self.close()
        else:
            msg.setText('Empty Field')
            msg.exec_()

    def get_status(self):
        return self.k, self.submitted


