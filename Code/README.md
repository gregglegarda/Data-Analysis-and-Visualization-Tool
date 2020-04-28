# Final-Project-Group7

Getting Started:
1. Download the dataset from kaggle (https://www.kaggle.com/sobhanmoosavi/us-accidents/download)
2. Make sure you can open the csv file or the pandas read wont work
3. It takes a while for the csv file to open/load so be patient
4. Once opened, move the dataset csv file to the Code folder
5. Run main.py (this runs all the other scripts as well)

Using the application:
1. Training: The first step in using the app is choose the number of samples, select a train/test ratio, and train the models.
2. Predicting: The second step is to predict severity level based on your chosen feature inputs.
3. Tabs: You can explore the app by clicking on the various tabs within the GUI. You can view an image of the current model, view the loadtime of the map, and view correlation plots, histograms, and scatterplots of the data.


Script Descriptions:

  1. "main.py" is the script you must run in order to launch the app. It pulls in all the other scripts in our repository and runs them as well.
  2. "eda_stats.py" creates the EDA graphs for the app, post-training
  3. "main_window.py" sets up the look and interactivity of GUI
    
  4. "map_load_time.py" creates a graph of the map load time based on the number of samples the user pulls from the dataset
  5. "map_view.py" sets up the map in our GUI
  6. "pop_up_entry.py" creates a pop-up window when running the KNN model; empty submission performs gridsearch for best k, otherwise integer submission sets k manually
  7. "pre_process.py" preprocesses the dataset in preparation for ingestion by our models. Column deletion, PCA, changing of data types, etc.
  8. "predict.py" enables severity prediction within the app.
  9. "train_model.py" houses the code for all of our machine learning models, and creates model images.
  
