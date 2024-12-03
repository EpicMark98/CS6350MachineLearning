# CS6350MachineLearning
This is a machine learning library developed by Mark VanDyke for CS5350/6350 at the University of Utah.

# Decision Tree
To learn decision trees, you first need to populate data-desc.txt, train.csv, and test.csv with the data description, training data, and test data, respectively. Then run "DecisionTree.py". The program will prompt the user for the heuristic to use when splitting data as well as the depth of the tree and whether or not to replace unknown values with the most common. The program will learn a decision tree print out the training and test error.

The decision tree learning process is divided into several functions that can be called from other Python scripts that wish to use decision tree learning as part of a larger algorithm.

# Ensemble Learning
To run any of the Ensemble Learning algorithms, you  first need to populate data-desc.txt, train.csv, and test.csv with the data description, training data, and test data, respectively. These files should be in the same directory as AdaBoost.py, Bagging.py, and RandomForest.py. Then you simply run the python script cooresponding to the algorithm you want, passing no arguments. The program will display some options and allow you as the user to choose the parameters to run different sections of the homework assignment.

# Linear Regression
To run Gradient Descent, you  first need to populate train.csv and test.csv with the training data and test data, respectively. These files should be in the same directory as GradientDescent.py. Then you simply run the python script with no arguments. The program will display some options and allow you as the user to choose whether you want the batch or stochastic version of the algorithm.

# Perceptron
To run Perceptron, you simply need to run the Perceptron.py script. Then you can choose which variant to run and it will train and test on the bank note data.

# SVM
To run SVM, you simply need to run the SVM.py script. Then you can choose which variant to run and it will train and test on the bank note data.

# Neural Networks
To run Neural Networks, you simply run the NeuralNetworks.py script. It then asks you to enter hyperparameter values and it will train and test on the bank note data.

# Project
The Project folder contains all the files used for the class project. I experimented with several different algorithms and eventually settled on AdaBoost but the other algorithms are still in the code even thought they are not used.
