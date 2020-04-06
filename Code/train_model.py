import os
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime


class train():
    def __init__(self, train_inputs):
        super(train, self).__init__()
        self.data = 0
        self.train_inputs = train_inputs

        self.data_processing()
        #self.plot_map_points()
        self.create_model()

        print("Training complete")
    def create_model(self):
        print("Creating model...")
        print("Data in model class is:",self.train_inputs)

        ####### PUT TRAINING ALGORITHM HERE
        train_split = int(self.train_inputs[1])
        test_split = int(self.train_inputs[2])




        print("Model created")
    def data_processing(self):
        #########-------------------------------------- DATA PROCESSING -------------------------------------- #########
        print("--------------------------DATA PROCESSING--------------------------")
        print("Processing sample size of:", self.train_inputs[0])

        datafile = "US_Accidents_Dec19.csv"

        try:
            from Code import pre_process
        except:
            print("import exception")

        data_instance = pre_process.data_frame(datafile, self.train_inputs)
        self.data = data_instance.create_dataframe()
        data_instance.cleanup_data()

        #########-------------------------------------- DATA ANALYSIS -------------------------------------- #########
        print("--------------------------DATA ANALYSIS--------------------------")
        try:
            import eda_stats
        except:
            print("import exception")
        data_analysis = eda_stats.eda(self.data)
        data_analysis.perform_eda()

    def get_map_data_points(self):
        #try:
            #import map_view
        #except:
            #print("import exception")

        # pass the data points in map
        #file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "map.html"))
        #mapinstance = map_view.map_webview(file_path, self.data)  # pass datapoints
        return self.data