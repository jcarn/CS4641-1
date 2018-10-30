import time
from termcolor import colored, cprint

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import UsefulFunctions


# Function to run the decision tree
def analyze(X, Y):
    # Spliting the dataset into train and test
    clf = AdaBoostClassifier(DecisionTreeClassifier(criterion="gini"))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)
    # using randomized search and k-fold to figure out the best set of parameters and training data
    max_depth = [i for i in range(1, 15)]
    min_samples_leaf = [i for i in range(1, 11)]
    param_grid = dict(base_estimator__max_depth=max_depth, base_estimator__min_samples_leaf=min_samples_leaf)
    # Starting timer here
    start_time = time.time()
    grid = RandomizedSearchCV(clf, param_grid, n_iter=5, scoring="accuracy")
    grid.fit(X_train, Y_train)
    # Done with timing here
    cprint("Training time: {0} \n".format(time.time() - start_time), 'blue')
    min_samples_graph = [result.mean_validation_score for result in grid.grid_scores_][1:11]
    best_params = grid.best_params_
    training_x_and_y = [X_train, Y_train]
    testing_x_and_y = [X_test, Y_test]
    graph_data = [min_samples_leaf, min_samples_graph]
    cprint("The best parameters were: {0}".format(best_params), 'red')
    return grid.best_estimator_, grid.best_score_, training_x_and_y, testing_x_and_y, graph_data


def calc_accuracy(y_test, y_pred):
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("Report: \n", classification_report(y_test, y_pred))
    print("Accuracy Score: {0:.2f}%".format(accuracy_score(y_test, y_pred) * 100))


# Driver code
def main():
    UsefulFunctions.warning()

    # Building Phase
    first_X, first_Y = UsefulFunctions.loadVehicleData()
    clf_first, first_training_score, first_training_data, first_testing_data, first_graph_data = analyze(first_X, first_Y)
    print("Boosting Training Score (first) After Cross Validation: {0:.2f}%".format(first_training_score * 100))
    UsefulFunctions.calc_accuracy(first_training_data[1], clf_first.predict(first_training_data[0]), first_testing_data[1], clf_first.predict(first_testing_data[0]))

    second_X, second_Y = UsefulFunctions.loadWineData()
    clf_second, second_training_score, second_training_data, second_testing_data, second_graph_data = analyze(second_X, second_Y)
    print("Boosting Training Score (second) After Cross Validation: {0:.2f}%".format(second_training_score * 100))
    UsefulFunctions.calc_accuracy(second_training_data[1], clf_second.predict(second_training_data[0]), second_testing_data[1], clf_second.predict(second_testing_data[0]))


if __name__ == '__main__':
    main()