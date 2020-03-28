import cv2
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QMainWindow, QGroupBox, QGridLayout,QPushButton, QMessageBox
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap



def runit(app):
    gui = open_camera(app)
    gui.show()
    run = app.exec_()
    return gui, run

def stop(run):
    sys.exit(run)

#### ---------------------------------------------------------------THIS PORTION COPIED-------------------------------------------------------------------------
class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
#### ---------------------------------------------------------------END OF THIS PORTION COPIED------------------------------------------------------------------------

class open_camera(QMainWindow):
    def __init__(self, app):
        self.app = app
        super().__init__()

        # ==================# MAIN WIDGET LAYOUT #==================#
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QGridLayout())
        self.widget.layout().setContentsMargins(20, 20, 20, 20)
        self.widget.layout().setSpacing(5)
        self.setWindowTitle("Employee Hour Logger")
        self.widget.layout().setColumnStretch(0, 1)
        self.widget.layout().setColumnStretch(1, 4)
        self.showMaximized()
        self.initUI()
        # ==================# END OF MAIN WIDGET LAYOUT #==================#

        # ==================# GROUP BOX 1 #==================#
        # Small group1 added to main widget
        self.GroupBox1 = QGroupBox("Face Tracker")
        layout1 = QGridLayout()
        self.GroupBox1.setLayout(layout1)
        layout1.setContentsMargins(5, 5, 5, 5)
        layout1.setSpacing(5)
        self.widget.layout().addWidget(self.GroupBox1, 0, 0,1,1)

        # ==================# GROUP BOX 2 #==================#
        # Small group1 added to main widget
        self.GroupBox2 = QGroupBox("Group box 2")
        layout2 = QGridLayout()
        self.GroupBox2.setLayout(layout2)
        layout2.setContentsMargins(5, 5, 5, 5)
        layout2.setSpacing(5)
        self.widget.layout().addWidget(self.GroupBox2, 0, 1, 1,3)

        # ==================# VIDEO FEED LABEL (IN GROUP BOX 1)#==================#
        #video feed is in the label and is added to group box 1
        self.label = QLabel(self)
        self.label.resize(20, 20)
        layout1.addWidget(self.label)

        # ==================# GROUP BOX 2 BUTTONS#==================#
        ##CLOCK IN BUTTON
        button_clock_in = QPushButton('Clock In')
        #button_clock_in.clicked.connect(self.logout_success)
        layout2.addWidget(button_clock_in, 0, 0)

        ##CLOCK IN BUTTON
        button_clock_out = QPushButton('Clock Out')
        #button_clock_out.clicked.connect(self.logout_success)
        layout2.addWidget(button_clock_out, 1, 0)


        # ==================# LOGOUT BUTTON#==================#
        ##LOGOUT BUTTON
        button_logout = QPushButton('Logout')
        button_logout.clicked.connect(self.logout_success)
        # layout2.addRow(button_logout)
        self.widget.layout().addWidget(button_logout, 3, 0, 1, 4)




################################################################################## FUNCTIONS ##########################################################################################
#==============================# VIDEO THREAD FUNCTION#==============================#
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        #connect thread to label with set image function for continuous feed
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.show()


# ==============================# LOGOUT FUNCTION#==============================#
    def logout_success(self):
        msg = QMessageBox()
        msg.setText('Logged out successful')
        msg.exec_()
        ## go to login screen
        self.close()