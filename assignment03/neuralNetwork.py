import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import time
import UsefulFunctions


def analyze(X, Y):
    clf = MLPClassifier(activation='relu')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)
    num_epochs = [i for i in range(500, 8000, 500)]
    param_grid = dict(max_iter=num_epochs)
    grid = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
    start = time.time()
    grid.fit(X_train, Y_train)
    end = time.time() - start
    grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
    # min_samples_graph = [result.mean_validation_score for result in grid.grid_scores_]
    # min_samples_graph = min_samples_graph[1:11]
    # print(min_samples_graph)
    testing_x_and_y = [X_test, Y_test]
    graph_data = [num_epochs, grid_mean_scores]
    return grid.best_estimator_, grid.best_score_, [X_test, Y_test], graph_data, end


def cal_accuracy(y_test, y_pred):
    # print("Confusion Matrix: \n", )
    return accuracy_score(y_test, y_pred)

# Driver code
def main():
    wine_X, wine_Y = UsefulFunctions.loadWineData()
    clf_wine, wine_training_score, wine_testing_data, wine_graph_data, wine_elapsed_time = analyze(wine_X, wine_Y)
    print("Neural Network Tree Training Score (Wine) After Cross Validation: {0}%".format(wine_training_score * 100))
    print("Neural Network Took (Wine) {0}s to Train".format(wine_elapsed_time))
    start = time.time()
    results = clf_wine.predict(wine_testing_data[0])
    end = time.time() - start
    print("Neural Network (Wine) Took {0}s to Test".format(end))
    # print(confusion_matrix(wine_testing_data[1], results))
    print(wine_graph_data[1])
    print("Neural Testing Score for Wine {0}%".format(cal_accuracy(wine_testing_data[1], results) * 100))

    # fig = plt.figure(200)
    # ax1 = plt.subplot(211)
    # ax1.plot(abalone_graph_data[0], abalone_graph_data[1])
    # ax1.set_xlabel("Number of Epochs")
    # ax1.set_ylabel("Cross Validated Accuracy Score")
    # ax1.set_title("Score vs Number of Epochs for Vehicle Data")

    # ax2 = plt.subplot(212)
    # ax2.plot(wine_graph_data[0], wine_graph_data[1])
    # ax2.set_xlabel("Number of Epochs")
    # ax2.set_ylabel("Cross Validated Accuracy Score")
    # ax2.set_title("Score vs Number of Epochs for Wine Data")
    # fig.tight_layout()
    # plt.show()


if __name__=='__main__':
    main()