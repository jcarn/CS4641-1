# Unsupervised Learning

### Requirements
Most of the code has been implemented in Weka, the only thing that I wrote was the neural network, and I have previously given running instructions in assignment 01.


### Data Files
All of the data files are online on the UCI Machine Learning Repository. The links are below:
1. https://archive.ics.uci.edu/ml/datasets/Wine+Quality
2. https://archive.ics.uci.edu/ml/datasets/Wine+Quality
I have also stored these files locally in the files/ subfolder. To generate the test/ train splits, I used Weka's RemovePercentage filter.
Some of the datasets had heavy modifications made to them, especially the ICA files and the clustered files, those are stored within the datasets folder.
All of the numbers that I got from running the algorithms are stored in the large Excel spreadsheet called results/AllResults.xslx


### Running the Files
Import the files into Weka, and then run the clustering and dimension reduction algorithms on them using the options provided either in the Weka CLI or the Weka GUI.

Run my Python code by typing in `python3 neuralNetwork.py` after modifying the UsefulFunctions.py file to read in the specific file. It might be a bit of a pain, but you need to modify the URL that is being read in, and the number of attributes to be parsed, which differs depending on the filtered dataset that is being parsed.
