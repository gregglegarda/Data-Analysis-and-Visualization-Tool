x=0
'''
    def logistic_regression(self, X_train, X_test, y_train, y_test):
        regressor = LogisticRegression(multi_class="multinomial",solver='saga')
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
        Z = regressor.predict(np.c_[xx.ravel(), yy.ravel()])

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
'''