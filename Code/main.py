###

#call the GUI window
import PyQt5
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5 import QtCore

#################################### CREATE THE APPLICATION
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
app = QApplication([])


################################### RUN HOME GUI###
import home_gui
homeapp,run = home_gui.runit(app)
succ, usertype = homeapp.login_button()
if succ != True:
    print("Home Closed.. Quitting App..")
    home_gui.stop(run)
else:
    print("Home Screen Closed")
    homeapp.close()

################# START BY RUNNING THE CAMERA ##################
if usertype == "Employee":
    print("Running Employee Terminal")
    import camera
    camapp,run = camera.runit(app)


#import geopandas as gpd
#shapefile = 'data/countries_110m/ne_110m_admin_0_countries.shp'
#Read shapefile using Geopandas
#gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]
#Rename columns.
#gdf.columns = ['country', 'country_code', 'geometry']
#gdf.head()

