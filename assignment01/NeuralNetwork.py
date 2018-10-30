# import time
# from termcolor import colored, cprint

# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.neural_network import MLPClassifier
# import UsefulFunctions


# def analyze(X, Y):
# 	clf = MLPClassifier(max_iter=500, activation='tanh', warm_start=True)
# 	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)
# 	momentums = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# 	learning_rates = [0.1, 0.3, 0.5]
# 	param_grid = dict(momentum=momentums, learning_rate_init=learning_rates)
# 	start_time = time.time()
# 	scores = []
# 	for i in range(1, 20):
# 		grid = RandomizedSearchCV(clf, param_grid, n_iter=10, scoring="accuracy")
# 		grid.fit(X_train, Y_train)
# 		scores.append(accuracy_score(Y_test, grid.predict(X_test)))
# 	cprint(("Training Time: {0}").format(time.time() - start_time), 'blue')
# 	best_params = grid.best_params_
# 	grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
# 	training_x_and_y = [X_train, Y_train]
# 	testing_x_and_y = [X_test, Y_test]
# 	# graph_data = [momentums, grid_mean_scores]
# 	graph_data = [[i for i in range(500, 10000, 500)], scores]
# 	cprint(("The best parameter was: {0}".format(best_params)), 'red')
# 	return grid.best_estimator_, grid.best_score_, training_x_and_y, testing_x_and_y, graph_data


# def calc_accuracy(y_test, y_pred):
#     print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
#     print("Report : \n", classification_report(y_test, y_pred))
#     print("Accuracy Score: {0:.2f}%".format(accuracy_score(y_test, y_pred) * 100))


# def graph_data(first_graph_data, second_graph_data):
# 	# fig = plt.figure(200)
# 	# ax1 = plt.subplot(211)
# 	# ax1.plot(first_graph_data[0], first_graph_data[1])
# 	# ax1.set_xlabel("Number of epochs")
# 	# ax1.set_ylabel("Accuracy Score")
# 	# ax1.set_title("Score vs Number of Epochs for Abalone Data")

# 	# ax2 = plt.subplot(212)
# 	# ax2.plot(second_graph_data[0], second_graph_data[1])
# 	# ax2.set_xlabel("Number of Epochs")
# 	# ax2.set_ylabel("Accuracy Score")
# 	# ax2.set_title("Score vs Number of Epochs for Wine Data")
# 	# fig.tight_layout()
# 	# plt.show()

# 	fig = plt.figure(200)
# 	ax1 = plt.subplot(211)
# 	ax1.plot(first_graph_data[0], first_graph_data[1], 'red')
# 	ax1.set_xlabel("Number of epochs")
# 	ax1.set_ylabel("Accuracy Score")
# 	ax1.set_title("Score vs Number of Epochs for Vehicle Data")

# 	ax2 = plt.subplot(212)
# 	ax2.plot(second_graph_data[0], second_graph_data[1], 'red')
# 	ax2.set_xlabel("Number of Epochs")
# 	ax2.set_ylabel("Accuracy Score")
# 	ax2.set_title("Score vs Number of Epochs for Wine Data")

# 	plt.legend(['Training Scores', 'Testing Scores'])
# 	fig.tight_layout()
# 	plt.show()


# def main():
# 	UsefulFunctions.warning()

# 	# Building Phase
# 	first_X, first_Y = UsefulFunctions.loadVehicleData()
# 	clf_first, first_training_score, first_training_data, first_testing_data, first_graph_data = analyze(first_X, first_Y)
# 	print("Neural Network Training Score (first) After Cross Validation: {0:.2f}%".format(first_training_score * 100))
# 	UsefulFunctions.calc_accuracy(first_training_data[1], clf_first.predict(first_training_data[0]), first_testing_data[1], clf_first.predict(first_testing_data[0]))

# 	second_X, second_Y = UsefulFunctions.loadWineData()
# 	clf_second, second_training_score, second_training_data, second_testing_data, second_graph_data = analyze(second_X, second_Y)
# 	print("\nNeural Network Training Score (second) After Cross Validation: {0:.2f}%".format(second_training_score * 100))
# 	UsefulFunctions.calc_accuracy(second_training_data[1], clf_second.predict(second_training_data[0]), second_testing_data[1], clf_second.predict(second_testing_data[0]))

# 	graph_data(first_graph_data, second_graph_data)


# if __name__ == '__main__':
# 	main()

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
    num_epochs = [i for i in range(500, 5000, 500)]
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
    abaloneX, abaloneY = UsefulFunctions.loadVehicleData()
    clf_abalone, abalone_training_score, abalone_testing_data, abalone_graph_data, abalone_elapsed_time = analyze(abaloneX, abaloneY)
    print("Neural Network Training Score (Abalone) After Cross Validation: {0}%".format(abalone_training_score * 100))
    print("Neural Network Took (Abalone) {0}s to Train".format(abalone_elapsed_time))
    start = time.time()
    results = clf_abalone.predict(abalone_testing_data[0])
    end = time.time() - start
    print("Neural Network (Abalone) Took {0}s to Test".format(end))
    print(confusion_matrix(abalone_testing_data[1], results))
    print("Neural Testing Score for Abalone {0}%".format(cal_accuracy(abalone_testing_data[1], results) * 100))


    wine_X, wine_Y = UsefulFunctions.loadWineData()
    clf_wine, wine_training_score, wine_testing_data, wine_graph_data, wine_elapsed_time = analyze(wine_X, wine_Y)
    print("Neural Network Tree Training Score (Wine) After Cross Validation: {0}%".format(wine_training_score * 100))
    print("Neural Network Took (Wine) {0}s to Train".format(wine_elapsed_time))
    start = time.time()
    results = clf_wine.predict(wine_testing_data[0])
    end = time.time() - start
    print("Neural Network (Wine) Took {0}s to Test".format(end))
    print(confusion_matrix(wine_testing_data[1], results))
    print("Neural Testing Score for Wine {0}%".format(cal_accuracy(wine_testing_data[1], results) * 100))

    fig = plt.figure(200)
    ax1 = plt.subplot(211)
    ax1.plot(abalone_graph_data[0], abalone_graph_data[1])
    ax1.set_xlabel("Number of Epochs")
    ax1.set_ylabel("Cross Validated Accuracy Score")
    ax1.set_title("Score vs Number of Epochs for Vehicle Data")

    ax2 = plt.subplot(212)
    ax2.plot(wine_graph_data[0], wine_graph_data[1])
    ax2.set_xlabel("Number of Epochs")
    ax2.set_ylabel("Cross Validated Accuracy Score")
    ax2.set_title("Score vs Number of Epochs for Wine Data")
    fig.tight_layout()
    plt.show()


if __name__=='__main__':
    main()