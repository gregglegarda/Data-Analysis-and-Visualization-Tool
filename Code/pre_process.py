
import pandas as pd
import numpy as np
import os



class data_frame():
      def __init__(self, datafile,samplesize):
            self.data = 0
            self.datafile = datafile
            self.samplesize = int(samplesize[0])
            super(data_frame, self).__init__()



      def create_dataframe(self):
            #########-------------------------------------- DATA FRAME -------------------------------------- #########
            #data = pd.read_csv("https://www.kaggle.com/sobhanmoosavi/us-accidents#US_Accidents_Dec19.csv")
            #data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_february_us_airport_traffic.csv')
            #print("--------------------------BUILD DATA FRAME--------------------------")
            print("Building Dataframe...")
            self.dataframe = pd.read_csv(self.datafile)
            # take random sample number of the 2 million rows
            self.data = self.dataframe.sample(self.samplesize)
            print("Dataframe Created.")
            return self.data

      def cleanup_data(self):
            #########-------------------------------------- DATA CLEANUP -------------------------------------- #########
            print("Cleaning Dataframe...")
            #delete uneccessary columns
            del self.data['ID']
            del self.data['Source']
            del self.data['TMC']
            del self.data['End_Lat']
            del self.data['End_Lng']
            del self.data['Country']

            #change datatypes and fill in nans
            self.data['Description'] = self.data['Description'].fillna(0)
            self.data['Street'] = self.data['Street'].fillna(0)
            self.data['City'] = self.data['City'].fillna(0)
            self.data['State'] = self.data['State'].fillna(0)
            self.data['Zipcode'] = self.data['Zipcode'].fillna(0)
            self.data['Description'] = np.where(pd.isnull(self.data['Description']),self.data['Description'],self.data['Description'].astype(str))
            self.data['Street'] = np.where(pd.isnull(self.data['Street']),self.data['Street'],self.data['Street'].astype(str))
            self.data['City'] = np.where(pd.isnull(self.data['City']),self.data['City'],self.data['City'].astype(str))
            self.data['State'] = np.where(pd.isnull(self.data['State']),self.data['State'],self.data['State'].astype(str))
            self.data['Zipcode'] = np.where(pd.isnull(self.data['Zipcode']),self.data['Zipcode'],self.data['Zipcode'].astype(str))


            #print results
            print("--------------------------COLUMNS--------------------------\n", self.data.columns)
            print("--------------------------DATA TYPES--------------------------:\n", self.data.dtypes)
            print("--------------------------HEAD--------------------------\n", self.data.head(10))
            print("--------------------------NUMBER OF SAMPLES--------------------------\n",
                  len(self.data),"Data points randomly selected from ", len(self.dataframe), "\n\n")

            self.data.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "pre_processed_data.csv")), index=False,
                      header=True)