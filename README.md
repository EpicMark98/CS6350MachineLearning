# CS6350MachineLearning
This is a machine learning library developed by Mark VanDyke for CS5350/6350 at the University of Utah.

# Decision Tree
To learn decision trees, you first need to populate data-desc.txt, train.csv, and test.csv with the data description, training data, and test data, respectively. Then run "DecisionTree.py". The program will prompt the user for the heuristic to use when splitting data as well as the depth of the tree and whether or not to replace unknown values with the most common. The program will learn a decision tree print out the training and test error.

The decision tree learning process is divided into several functions that can be called from other Python scripts that wish to use decision tree learning as part of a larger algorithm.
