import csv
import pandas as pd
import random
import time
from termcolor import colored, cprint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings


def loadWineData():
    balance_data = pd.read_csv('datasets/winequality-rs-cluster-em.csv',
        sep= ',', header=1, names=['k1','k2','k3','k4','k5','k6', 'Cluster','quality'])
    mapping = {'cluster0': 0, 'cluster1': 1, 'cluster2': 2, 'cluster3': 3, 'cluster4': 4, 'cluster5': 5}
    balance_data.Cluster = balance_data.Cluster.replace(mapping)
    X, Y = balance_data.values[:, 0:7], balance_data.values[:, 7]
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
