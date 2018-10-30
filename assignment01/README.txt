# README

### Initial Steps
The first thing to be done is to install all of the required packages. To do this, cd into the root directory, and then type:
```bash
pip install -r requirements.txt
```
This code is written in Python 3, so please run this code in Python 3. It might work in Python 2, but I have not checked, so I am not responsible for any bugs that might ensue.

### Datasets that are being used
For this assignment, I used the vehicles and white wine quality datasets from the UCI Machine Learning repository. Links to the two datasets are below:
* Vehicle dataset: https://archive.ics.uci.edu/ml/datasets/Statlog+%28Vehicle+Silhouettes%29
* Wine dataset: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
There is functionality built into the code to run other datasets, if you want to. Those datasets are the abalone dataset, the poker dataset and the cancer dataset, also from the UCI Machine Learning repository. Look at the `UsefulFunctions.py` file to find out more.

### Running Code
To run any of this code, simply type `python NAME_OF_FILE.py`.
1. Decision Tree: `python DecisionTree.py`
2. Boosting: `python Boosting.py`
3. k Nearest Neighbors: `python kNN.py`
	* There are two options to run this, running it normally will not outpt training and testing accuracy curves. If you want this, run
	```v
	python kNN.py --curves
	```
4. Neural Network: `python NeuralNetwork.py`
5. Support Vector Machines: `python SupportVectorMachines.py`
