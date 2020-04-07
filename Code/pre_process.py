
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
            del self.data['Timezone']
            del self.data['Airport_Code']
            del self.data['Weather_Timestamp']
            del self.data['Wind_Direction']

            del self.data['Weather_Condition']




            #########-------------------------------------- FILL MISSING VALUES -------------------------------------- #########

            # fill other predict attributes with means of columns
            self.data['Start_Lat'] = self.data['Start_Lat'].fillna(self.data['Start_Lat'].mean())
            self.data['Start_Lng'] = self.data['Start_Lng'].fillna(self.data['Start_Lng'].mean())
            self.data['Distance(mi)'] = self.data['Distance(mi)'].fillna(self.data['Distance(mi)'].mean())
            self.data['Number'] = self.data['Number'].fillna(self.data['Number'].mean())
            self.data['Temperature(F)'] = self.data['Temperature(F)'].fillna(self.data['Temperature(F)'].mean())
            self.data['Wind_Chill(F)'] = self.data['Wind_Chill(F)'].fillna(self.data['Wind_Chill(F)'].mean())
            self.data['Humidity(%)'] = self.data['Humidity(%)'].fillna(self.data['Humidity(%)'].mean())
            self.data['Pressure(in)'] = self.data['Pressure(in)'].fillna(self.data['Pressure(in)'].mean())
            self.data['Visibility(mi)'] = self.data['Visibility(mi)'].fillna(self.data['Visibility(mi)'].mean())
            self.data['Wind_Speed(mph)'] = self.data['Wind_Speed(mph)'].fillna(self.data['Wind_Speed(mph)'].mean())
            self.data['Precipitation(in)'] = self.data['Precipitation(in)'].fillna(self.data['Precipitation(in)'].mean())
            self.data['Severity'] = self.data['Severity'].fillna(self.data['Severity'].mean())

            # fill address with filler
            self.data['Description'] = self.data['Description'].fillna(0)
            self.data['Street'] = self.data['Street'].fillna(0)
            self.data['City'] = self.data['City'].fillna(0)
            self.data['State'] = self.data['State'].fillna(0)
            self.data['Zipcode'] = self.data['Zipcode'].fillna(0)

            #########-------------------------------------- CHANGE DATA TYPES -------------------------------------- #########
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
            print("--------------------------INFO AFTER CLEANING--------------------------\n", self.data.info())
            self.data.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "pre_processed_data.csv")), index=False,
                      header=True)