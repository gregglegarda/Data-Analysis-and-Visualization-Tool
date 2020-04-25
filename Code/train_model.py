
import pydotplus
import collections
from sklearn.tree import plot_tree
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl


class train():
    def __init__(self, train_inputs, k_value,app ):
        super(train, self).__init__()
       #attributes
        self.app = app
        self.accuracy = 0
        self.data = 0
        self.train_inputs = train_inputs
        self.model_algorithm = 0
        self.k_value = int(k_value)

        #functions
        self.data_processing()
        self.create_model()

        print("Training complete")
####==================================   CREATE MODEL FUNCTION ====================================############

    def create_model(self):
        print("Creating model...")
        print("Data in model class is:",self.train_inputs)

        #######  TRAIN AND SPLIT #######
        train_split = int(self.train_inputs[1][0] + self.train_inputs[1][1] ) #only take the first two digits since it has a %
        test_split = (100 - train_split)/100
        model_algorithim = self.train_inputs[2]

        # make table for preprocessing
        column_names = ["Distance(mi)", "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
                        "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
                        "Severity"]

        X = [list(self.data["Distance(mi)"]), list(self.data["Temperature(F)"]),
             list(self.data["Wind_Chill(F)"]), list(self.data["Humidity(%)"]), list(self.data["Pressure(in)"]),
             list(self.data["Visibility(mi)"]), list(self.data["Wind_Speed(mph)"]), list(self.data["Precipitation(in)"])]
        X = np.transpose(X)
        y = list(self.data["Severity"])


        #perform PCA before splitting,
        if self.train_inputs[2] == "KNN" or self.train_inputs[2] == "Logistic Regression":
            X = PCA(n_components=2).fit_transform(X)
            print("PCA PERFORMED")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=0)


        #########   PICKED MODEL GOES HERE   ###
        if self.train_inputs[2] == "Decision Trees":
            self.decision_tree(X_train, X_test, y_train, y_test)
        elif self.train_inputs[2] == "Random Forest":
            self.random_forest(X_train, X_test, y_train, y_test)
        elif self.train_inputs[2] == "Logistic Regression":
            self.logistic_regression(X_train, X_test, y_train, y_test)
        elif self.train_inputs[2] == "KNN":
            self.knn_classifier(X_train, X_test, y_train, y_test)
        elif self.train_inputs[2] == "SVM":
            self.svm_classifier(X_train, X_test, y_train, y_test)
        elif self.train_inputs[2] == "Naive Bayes":
            self.naive_bayes(X_train, X_test, y_train, y_test)

###================================ MACHINE LEARNING FUNCTIONS =====================================###

    ##========================  DECISION TREE ===================###
    def decision_tree(self, X_train, X_test, y_train, y_test):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))

        # Build decision tree model
        # visualize tree train




        # Decision tree gridsearch
        clf = tree.DecisionTreeClassifier(random_state=23)
        clf.fit(X_train_std,y_train)
        from sklearn.model_selection import \
            GridSearchCV  # https://medium.com/@haydar_ai/learning-data-science-day-22-cross-validation-and-parameter-tuning-b14bcbc6b012

        parameter_grid = {'criterion': ['gini', 'entropy'],  # provides better parameters for model above
                          'splitter': ['best', 'random'],
                          'max_depth': [1, 2, 3, 4, 5],
                          'max_features': [1, 2, 3, 4]}

        grid_search = GridSearchCV(clf, param_grid=parameter_grid, cv=10)

        grid_result = grid_search.fit(X_train_std, y_train)
        best_params = grid_result.best_params_
        print(best_params)
        best_clf = tree.DecisionTreeClassifier(criterion=best_params['criterion'],splitter=best_params['splitter'],
                                         max_depth=best_params['max_depth'], max_features=best_params['max_features'],random_state=23)

        best_clf.fit(X_train_std,y_train)

        print('Model created')

        # Predict test from train
        y_pred = best_clf.predict(X_test_std)
        print("shape of y_pred:", X_test_std.shape)
        print("y_pred:", X_test_std)
        acc_score = accuracy_score(y_test, y_pred)
        self.accuracy = (acc_score * 100).round(2)
        # Accuracy
        print('DT Accuracy:', acc_score)
        print('DT Accuracy:', self.accuracy)
        self.model_algorithm = best_clf
        print("train model is:", best_clf)


        feature_names = ["Distance(mi)", "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
                        "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]

        ####VISUALIZE PLOT THE MODEL AND SAVE
        dot_data = tree.export_graphviz(clf,
                                         feature_names=feature_names,
                                         out_file=None,
                                         filled=True,
                                         rounded=True,
                                         )
        graph = pydotplus.graph_from_dot_data(dot_data)
        colors = ('grey', 'lightgray', 'yellow')# turquoiseinvert rgb(191,31,47)... orange invert rgb(0,90,255)#005AFF
        edges = collections.defaultdict(list)
        for edge in graph.get_edge_list():
             edges[edge.get_source()].append(int(edge.get_destination()))
        for edge in edges:
             edges[edge].sort()
             for i in range(2):
                 dest = graph.get_node(str(edges[edge][i]))[0]
                 dest.set_fillcolor(colors[i])
        graph.write_png('model_image.png')
        self.dark_mode_png()


        ####### alternate tree with dark background
        #plt.style.use('dark_background')
        #mpl.rcParams['text.color'] = 'black'
        #fig, ax = plt.subplots(figsize=(30, 30), facecolor='k')
        #plot_tree(clf, rotate=True, ax=ax, feature_names= feature_names, max_depth= 3)
        #plt.savefig('model_image.png',facecolor='#1a1a1a')#,transparent=True,)

    ##========================  RANDOM FOREST ===================###
    def random_forest(self, X_train, X_test, y_train, y_test):
        rf = RandomForestClassifier(n_estimators=100, random_state=23, verbose=3, n_jobs=-1)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        rf.fit(X_train_std,y_train)
        # https://stackoverflow.com/questions/43640546/how-to-make-randomforestclassifier-faster
        # random forest GridSearch
        from sklearn.model_selection import \
            GridSearchCV  # https://medium.com/@haydar_ai/learning-data-science-day-22-cross-validation-and-parameter-tuning-b14bcbc6b012
        parameter_grid = {'criterion': ['gini', 'entropy'],  # provides better parameters for model above
                          'max_depth': [1, 2, 3, 4, 5],
                          'max_features': [1, 2, 3, 4]}
        CV_rf = GridSearchCV(rf,param_grid=parameter_grid,cv=10)
        grid_result = CV_rf.fit(X_train_std,y_train)
        best_params = grid_result.best_params_
        print(best_params)

        best_rf = RandomForestClassifier(n_estimators=100,criterion=best_params['criterion'],
                                         max_depth=best_params['max_depth'],max_features=best_params['max_features'],
                                         verbose=3,n_jobs=-1,random_state=23)
        best_rf.fit(X_train_std,y_train)
        # https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
        predictions = best_rf.predict(X_test_std)  # Calculate the absolute errors

        # Accuracy
        accuracy = accuracy_score(y_test, predictions)*100
        print('Random Forest Model Accuracy:', round(accuracy, 2), '%.')

        self.accuracy = round(accuracy, 2)
        self.model_algorithm = best_rf
        print("train model is:", best_rf)

        #PICK A TREE FROM THE RANDOM FOREST
        tree = best_rf.estimators_[5]



        ####PLOT THE MODEL #style is dark background plot_tree different look from decision tree
        plt.figure()
        feature_names = ["Distance(mi)", "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
                         "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]
        plt.style.use('dark_background')
        mpl.rcParams['text.color'] = 'black'
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='k')
        fig.suptitle('SINGLE TREE FROM RANDOM FOREST', fontsize=12, color='w')
        plot_tree(tree, rotate=True, ax=ax, feature_names= feature_names, max_depth= 3 )
        plt.savefig('model_image.png',facecolor='#1a1a1a')#,transparent=True,)
        #plt.savefig('model_image.png',facecolor='#1a1a1a',transparent=True,)
        plt.close()
    ##========================  LOGISTIC REGRESSION ===================###
    def logistic_regression(self, X_train, X_test, y_train, y_test):
        regressor = LogisticRegression(multi_class="multinomial")
        regressor.fit(X_train, y_train)  # training the algorithm
        predictions = regressor.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, predictions)*100
        reg_score = regressor.score(X_test,y_test)
        print('Logistic Regression Model Accuracy:', round(accuracy, 2), '%.')
        print('Logistic Regression Score:', round(reg_score, 2), '%.')


        self.accuracy = round(accuracy, 2)
        self.model_algorithm = regressor
        print("train model is:", regressor)


        self.accuracy = round(accuracy, 2)
        self.model_algorithm = best_regression
        print("train model is:", best_regression)


        ####PLOT THE MODEL

        plt.figure()
        title_obj = plt.title('LOGISTIC REGRESSION')
        plt.setp(title_obj, color='w')

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
        y_min, y_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
        h = .02  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = best_regression.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(4, 3))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

        # Plot also the training points
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.Paired)
        x_label = plt.xlabel('Principal Component 1')
        y_label = plt.ylabel('Principal Component 2')
        plt.setp(x_label, color='w')
        plt.setp(y_label, color='w')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

        plt.savefig('model_image.png', facecolor='#1a1a1a', transparent=True, )
        plt.close()


    ##========================  K_NEAREST NEIGHBORS  ===================###
    def knn_classifier(self, X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA):
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_jobs=-1)
        # KNN  GridSearch
        from sklearn.model_selection import \
            GridSearchCV  # https://medium.com/@haydar_ai/learning-data-science-day-22-cross-validation-and-parameter-tuning-b14bcbc6b012
        parameter_grid = {'metric': ['minkowski','manhattan','euclidean'],
                          'leaf_size': [1,2,3,4,5],
                          'weights': ['uniform', 'distance'],
                          'n_neighbors': [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49]}
        CV_knn = GridSearchCV(neigh, param_grid=parameter_grid, cv=10)
        grid_result = CV_knn.fit(X_train_PCA, y_train_PCA)
        best_params = grid_result.best_params_
        print(grid_result.best_params_)

        best_knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'],n_jobs=-1, leaf_size=best_params['leaf_size'],
                                         metric=best_params['metric'],weights=best_params['weights'])
        best_knn.fit(X_train_PCA, y_train_PCA)

        # https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
        predictions = best_knn.predict(X_test_PCA)  # Calculate the absolute errors

        # Accuracy
        accuracy = accuracy_score(y_test_PCA, predictions)*100
        print('KNN Model Accuracy:', round(accuracy, 2), '%.')

        self.accuracy = round(accuracy, 2)
        self.model_algorithm = best_knn
        print("train model is:", best_knn)

        ####================create a graph for KNN curve to find optimal elbow
        # from 1 to length of training samples (usually 70%), step size is divided by 100
        #k_range = range(1, len(X_train_PCA)-1, int(int(self.train_inputs[0])/100))

        k_range = range(1, 50, 2)
        scores = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k,n_jobs=-1,metric='minkowski',weights='uniform',leaf_size=1)
            knn.fit(X_train_PCA, y_train_PCA)
            y_pred = knn.predict(X_test_PCA)
            #scores.append(accuracy_score(y_test_PCA, y_pred))
            scores.append(np.mean(y_pred != y_test_PCA))
            print(k, k_range)


        #PLOT THE MODEL

        plt.figure()
        plt.clf()
        plt.plot(k_range, scores)
        title_obj = plt.title('KNN')
        plt.setp(title_obj, color='w')
        x_label = plt.xlabel('Value of K for KNN')
        y_label = plt.ylabel('Error Rate')
        plt.setp(x_label, color='w')
        plt.setp(y_label, color='w')


        #model look
        plt.tick_params(axis='both', colors='white', labelsize = 6)
        ax = plt.gca()
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.spines['left'].set_color('w')
        plt.savefig('model_image.png',facecolor='#1a1a1a',transparent=True,)
        plt.close()

    ##========================  SUPORT VECTOR MACHINE ===================###
    def svm_classifier(self, X_train, X_test, y_train, y_test):
        from sklearn.svm import LinearSVC
        from sklearn.model_selection import \
            GridSearchCV  # https://medium.com/@haydar_ai/learning-data-science-day-22-cross-validation-and-parameter-tuning-b14bcbc6b012
        svc = LinearSVC(random_state=0, tol=1e-5)
        parameter_grid = {'max_iter': [1000,5000,10000],
                          'C': [1.0,2.0,3.0,4.0,5.0]}
        CV_svc = GridSearchCV(svc, param_grid=parameter_grid, cv=10)
        grid_result = CV_svc.fit(X_train, y_train)
        best_params = grid_result.best_params_
        print(grid_result.best_params_)

        best_svc = LinearSVC(dual=False, C=best_params['C'],max_iter=best_params['max_iter'])
        best_svc.fit(X_train, y_train)

        predictions = best_svc.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, predictions) * 100
        print('SVM Accuracy:', round(accuracy, 2), '%.')

        self.accuracy = round(accuracy, 2)
        self.model_algorithm = best_svc
        print("train model is:", best_svc)

        ####PLOT THE MODEL
        plt.figure()
        title_obj = plt.title('SVM')
        plt.setp(title_obj, color='w')
        #plt.clf()
        plt.savefig('model_image.png',facecolor='#1a1a1a',transparent=True,)
        plt.close()

    ##========================  NAIVE BAYES  ===================###
    def naive_bayes(self, X_train, X_test, y_train, y_test):
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, predictions) * 100
        print('Naive Bayes Model Accuracy:', round(accuracy, 2), '%.')

        self.accuracy = round(accuracy, 2)
        self.model_algorithm = clf
        print("train model is:", clf)

        ####PLOT THE MODEL
        plt.figure()
        title_obj = plt.title('NAIVE BAYES')
        plt.setp(title_obj, color='w')
        #plt.clf()
        plt.savefig('model_image.png',facecolor='#1a1a1a',transparent=True,)
        plt.close()

#########-------------------------------------- DATA PROCESSING AND ANALYSIS -------------------------------------- #########
    def data_processing(self):
        ######### DATA PROCESSING  #########
        print("--------------------------DATA PROCESSING--------------------------")
        print("Processing sample size of:", self.train_inputs[0])

        datafile = "US_Accidents_Dec19.csv"

        try:
            import pre_process
        except:
            print("import exception")

        data_instance = pre_process.data_frame(datafile, self.train_inputs)
        self.data = data_instance.create_dataframe()
        data_instance.cleanup_data()

        ######### DATA ANALYSIS  #########
        print("--------------------------DATA ANALYSIS--------------------------")
        try:
            import eda_stats
        except:
            print("import exception")
        data_analysis = eda_stats.eda(self.data)
        data_analysis.perform_eda()
###============================== GET DATA FUNCTIONS ===============================###########
    def get_map_data_points(self):
        return self.data
    def get_model_accuracy(self):
        return self.accuracy
    def get_model(self):
        return self.model_algorithm
#### change image colors
    def dark_mode_png(self):
        from PIL import ImageOps
        from PIL import Image
        rgba_image = Image.open('model_image.png')
        print("rgba",np.shape(rgba_image))

        rgba_image.load()
        background = Image.new("RGB", rgba_image.size, (255, 255, 255))

        try:
            self.app.processEvents()
            print("rgba", np.shape(rgba_image))
            background.paste(rgba_image, mask=rgba_image.split()[3])
            background.save("model_image.png", "PNG", quality=100)
            rgb_image = Image.open("model_image.png")
            im = ImageOps.invert(rgb_image)
            print("im", np.shape(rgb_image))
        except:
            print("tuple error")
            try:
                self.app.processEvents()
                im = ImageOps.invert(rgba_image) ##rgba really rgb at this point
                print("im", np.shape(rgba_image))
            except:
                im=rgba_image
                print("color invert error")




        im = im.convert('RGBA')
        data = np.array(im)  # "data" is a height x width x 4 numpy array
        red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability

        black_areas = (red == 0) & (blue == 0) & (green == 0)  ###255,255,255 is white
        data[..., :-1][black_areas.T] = (26, 26, 26)  # Transpose back needed

        im2 = Image.fromarray(data)
        im2.save('model_image.png')
        #im2.show()
