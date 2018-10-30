import csv
import pandas as pd
import random
import time
from termcolor import colored, cprint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings

# Loads data from a file into a training set and a test set
# X is the features, Y is the classes
def loadAbaloneData():
    balance_data = pd.read_csv(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',sep=',', header=None, names=["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera Weight", "shell weight", "rings"])
    mapping = {'M': 1, 'I': 2, 'F': 3}
    balance_data.sex = balance_data.sex.replace(mapping)
    X, Y = balance_data.values[:, 0:8], balance_data.values[:, 8]
    return X, Y


def loadWineData():
    balance_data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
    sep= ';', header=0)
    X, Y = balance_data.values[:, 0:11], balance_data.values[:, 11]
    return X, Y

def loadPokerData():
    balance_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
        sep=',', header=None, names=["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "class"])
    X, Y = balance_data.values[:, 0:10], balance_data.values[:, 10]
    return X, Y


def loadCancerData():
    balance_data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data',
        sep=',', header=None, names=["ID", "clumpThickness", "cellSizeUniformity", "cellShapeUniformity", "marginalAdhesion", "cellSize", "nuclei", "chromatin", "normalNuclei", "mitoses", "class"])
    X, Y = balance_data.values[:, 1:10], balance_data.values[:, 10]
    return X, Y


def loadVehicleData():
    balance_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaa.dat',
        sep=' ', header=None, names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'type'])
    balance_data.append(pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xab.dat',
        sep=' ', header=None, names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'type']))
    balance_data.append(pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xac.dat',
        sep=' ', header=None, names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'type']))
    balance_data.append(pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xad.dat',
        sep=' ', header=None, names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'type']))
    balance_data.append(pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xae.dat',
        sep=' ', header=None, names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'type']))
    balance_data.append(pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaf.dat',
        sep=' ', header=None, names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'type']))
    balance_data.append(pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xag.dat',
        sep=' ', header=None, names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'type']))
    balance_data.append(pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xah.dat',
        sep=' ', header=None, names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'type']))
    balance_data.append(pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xai.dat',
        sep=' ', header=None, names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'type']))
    mapping = {"opel": 1, "saab": 2, "bus": 3, "van": 4}
    balance_data.q = balance_data.q.replace(mapping)
    X, Y = balance_data.values[:, 0: 16], balance_data.values[:, 16]
    return X, Y


def calc_accuracy(y_train, y_train_pred, y_test, y_test_pred):
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_test_pred))
    print("Report: \n", classification_report(y_test, y_test_pred))
    start_time = time.time()
    print("Train Accuracy Score: ", accuracy_score(y_train, y_train_pred))
    print("Test Accuracy Score: ", accuracy_score(y_test, y_test_pred))
    cprint("Testing time: {0} \n".format(time.time() - start_time), 'blue')


def warning():
    warnings.filterwarnings("ignore")


# For exporting data to a clearly formatted CSV for report
def exportData(filename, columns, data):
	with open(filename, 'a') as f:
		f.write(",".join(columns))
		f.write("\n")
		for line in data:
			f.write(",".join(columns))
			f.write("\n")
