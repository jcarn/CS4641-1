import time
from termcolor import colored, cprint
import sys

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import UsefulFunctions

# Function to run the decision tree
def analyze(X, Y, numNeighb=30):
	# Spliting the dataset into train and test
	knn = KNeighborsClassifier(weights="distance")
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state = 100)
	k_range = [i for i in range(1, numNeighb)]
	param_grid = dict(n_neighbors=k_range)
	# Starting timing here
	start_time = time.time()
	grid = GridSearchCV(knn, param_grid, cv=3, scoring="accuracy")
	grid.fit(X_train, Y_train)
	# Printing out how long it took here
	cprint("Training time: {0}".format(time.time() - start_time), 'blue')
	scores = grid.grid_scores_
	grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
	graph_data = [k_range, grid_mean_scores]
	best_score = grid.best_score_
	best_params = grid.best_params_
	best_model = grid.best_estimator_
	training_x_and_y = [X_train, Y_train]
	testing_x_and_y = [X_test, Y_test]
	cprint("The most optimal number of neighbors was: {0}".format(best_params), 'red')
	return best_model, best_score, training_x_and_y, testing_x_and_y, graph_data


def analyzePerNeighbor(X, Y, numNeighb=30):
	trainingScores = []
	testingScores = []
	kNeighbors = [i for i in range(1, numNeighb)]
	for i in range(1, numNeighb):
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state = 100)
		knn = KNeighborsClassifier(n_neighbors=i, weights='distance')
		knn.fit(X_train, Y_train)
		trainingScores.append(knn.score(X_train, Y_train))
		testingScores.append(knn.score(X_test, Y_test))
	return [trainingScores, testingScores, kNeighbors]


# For graphing original CV stuff
def graphData(abalone_graph_data, wine_graph_data):
	# Prints out data to a CSV
	# f = open('results/knn.csv', 'a')
	# f.write("! Abalone Dataset\n")
	# for i in range(len(abalone_graph_data[0])):
	#     f.write(str(abalone_graph_data[0][i]))
	#     f.write(", ")
	#     f.write(str(abalone_graph_data[1][i]))
	#     f.write("\n")
	# f.write("\n!Wine Dataset\n")
	# for i in range(len(wine_graph_data[0])):
	#     f.write(str(wine_graph_data[0][i]))
	#     f.write(", ")
	#     f.write(str(wine_graph_data[1][i]))
	#     f.write("\n")
	# f.close()

	# Graphs out CV stuff
	fig = plt.figure(200)
	ax1 = plt.subplot(211)
	ax1.plot(abalone_graph_data[0], abalone_graph_data[1], 'black')
	ax1.set_xlabel("Number of Neighbors")
	ax1.set_ylabel("Cross Validated Accuracy Score")
	ax1.set_title("Score vs Number of Neighbors for Abalone Data")

	ax2 = plt.subplot(212)
	ax2.plot(wine_graph_data[0], wine_graph_data[1], 'black')
	ax2.set_xlabel("Number of Neighbors")
	ax2.set_ylabel("Cross Validated Accuracy Score")
	ax2.set_title("Score vs Number of Neighbors for Wine Data")
	fig.tight_layout()
	plt.show()


# Does learning curves
def graphDataCurves(abalone_graph_data, wine_graph_data):
	fig = plt.figure(200)
	ax1 = plt.subplot(211)
	ax1.plot(abalone_graph_data[-1], abalone_graph_data[0], 'red')
	ax1.plot(abalone_graph_data[-1], abalone_graph_data[1], 'green')
	ax1.set_xlabel("Number of Neighbors")
	ax1.set_ylabel("Accurately Identified")
	ax1.set_title("Testing and Training Scores for Vehicle")

	ax2 = plt.subplot(212)
	ax2.plot(wine_graph_data[-1], wine_graph_data[0], 'red')
	ax2.plot(wine_graph_data[-1], wine_graph_data[1], 'green')
	ax2.set_xlabel("Number of Neighbors")
	ax2.set_ylabel("Accurately Identified")
	ax2.set_title("Testing and Training Scores for Wine")
	fig.tight_layout()

	plt.legend(['Training Scores', 'Testing Scores'])
	plt.show()

def mainCurves():
	UsefulFunctions.warning()

	first_X, first_Y = UsefulFunctions.loadVehicleData()
	first_graph_data = analyzePerNeighbor(first_X, first_Y)

	second_X, second_Y = UsefulFunctions.loadWineData()
	second_graph_data = analyzePerNeighbor(second_X, second_Y)

	graphDataCurves(first_graph_data, second_graph_data)


# Driver code
def main():
	UsefulFunctions.warning()

	# Building Phase
	first_X, first_Y = UsefulFunctions.loadVehicleData()
	clf_first, first_training_score, first_training_data, first_testing_data, first_graph_data = analyze(first_X, first_Y, 40)
	print("kNN Training Score (first) After Cross Validation: {0:.2f}%".format(first_training_score * 100))
	UsefulFunctions.calc_accuracy(first_training_data[1], clf_first.predict(first_training_data[0]), first_testing_data[1], clf_first.predict(first_testing_data[0]))

	second_X, second_Y = UsefulFunctions.loadWineData()
	clf_second, second_training_score, second_training_data, second_testing_data, second_graph_data = analyze(second_X, second_Y)
	print("kNN Training Score (second) After GridSearch Cross Validation: {0:.2f}%".format(second_training_score * 100))
	UsefulFunctions.calc_accuracy(second_training_data[1], clf_second.predict(second_training_data[0]), second_testing_data[1], clf_second.predict(second_testing_data[0]))

	graphData(first_graph_data, second_graph_data)


if __name__ == '__main__':
	if len(sys.argv) > 1:
		if sys.argv[1] == '--curves':
			mainCurves()
		else:
			main()
	else:
		main()