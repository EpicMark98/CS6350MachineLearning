
import numpy as np
import matplotlib.pyplot as plt
import random

# Loads the data into a list of numpy arrays.
def load_examples(filename = "train.csv"):
    dataset = []
    with open(filename, 'r') as f:
        # For each example
        for line in f:
            # Get the attribute values and add them to a numpy array
            stringArray = line.strip().split(',')
            currItem = np.array([1] + [float(x) for x in stringArray])  # Convert strings and add 1 for bias

            # Replace 0 labels with -1
            if currItem[-1] == 0:
                currItem[-1] = -1

            # Add the current item to the dataset
            dataset.append(currItem)
    return dataset

# Gets the prediction for standard perceptron
def get_prediction(x, W):
    return 1 if np.dot(x, W) >= 0 else -1

# Calculates the cost for standard perceptron
def calc_error(S, W):
    totalError = 0
    for ex in S:
        # Split the example into x and y
        x_i = ex[:-1]
        y_i = ex[-1]

        # Get prediction and calculate error
        pred = get_prediction(x_i, W)
        if y_i != pred:
            totalError += 1

    return totalError / len(S)

# Gets the prediction
def get_prediction_voted(x, w_list, count_list):
    totalVote = 0;
    for i in range(len(w_list)):
        totalVote += count_list[i] * get_prediction(x, w_list[i])
    return 1 if totalVote >= 0 else -1

# Calculates the cost
def calc_error_voted(S, w_list, count_list):
    totalError = 0
    for ex in S:
        # Split the example into x and y
        x_i = ex[:-1]
        y_i = ex[-1]

        # Get prediction and calculate error
        pred = get_prediction_voted(x_i, w_list, count_list)
        if y_i != pred:
            totalError += 1

    return totalError / len(S)

# Runs the standard perceptron algorithm
def perceptron(train, r, epochs):
    # Initialize W
    W = np.array([0]*(len(train[0])-1))

    # Loop until the error stops changing
    for e in range(epochs):
        np.random.shuffle(train)
        for ex in train:
            x_i = ex[:-1]
            y_i = ex[-1]

            # Update the weight vector
            pred = get_prediction(x_i, W)
            if pred != y_i:
                W = W + r * x_i * y_i

    return W

# Runs the voted perceptron algorithm
def voted_perceptron(train, r, epochs):
    # Initialize W
    W = np.array([0]*(len(train[0])-1))
    w_list = []
    count_list = []
    count = 1

    # Loop until the error stops changing
    for e in range(epochs):
        np.random.shuffle(train)
        for ex in train:
            x_i = ex[:-1]
            y_i = ex[-1]

            # Update the weight vector
            pred = get_prediction(x_i, W)
            if pred != y_i:
                w_list.append(W)
                count_list.append(count)
                count = 1
                W = W + r * x_i * y_i
            else:
                count += 1

    return w_list, count_list
# Runs the average perceptron algorithm
def average_perceptron(train, r, epochs):
    # Initialize W
    W = np.array([0]*(len(train[0])-1))
    A = np.array([0]*(len(train[0])-1))

    # Loop until the error stops changing
    for e in range(epochs):
        np.random.shuffle(train)
        for ex in train:
            x_i = ex[:-1]
            y_i = ex[-1]

            # Update the weight vector
            pred = get_prediction(x_i, W)
            if pred != y_i:
                W = W + r * x_i * y_i

            # Update average
            A = np.add(A, W)

    return A

def WeightToStr(w):
    str = "["
    for v in w:
        str += "{:.2f}".format(v) + ", "
    str = str[:-2]
    str += "]"
    return str

# Wrapper for main
def main(choice):
    # Load examples
    trainingData = load_examples()
    testData = load_examples("test.csv")

    # Run perceptron
    if choice == 1:
        finalW = perceptron(trainingData, 0.2, 10)

        # Calculate test error
        error = calc_error(testData, finalW)
        print("Final W: " + str(finalW))
        print("Test error is " + "{:.4f}".format(error))
    elif choice == 2:
        w_list, count_list = voted_perceptron(trainingData, 0.2, 10)

        # Calculate test error
        error = calc_error_voted(testData, w_list, count_list)
        for i in range(len(w_list)):
            print("Count: " + str(count_list[i]) + "     Weights: " + WeightToStr(w_list[i]))
        print("Test error is " + "{:.4f}".format(error))
    else:
        finalW = average_perceptron(trainingData, 0.2, 10)

        # Calculate test error
        error = calc_error(testData, finalW)
        print("Final W: " + str(finalW))
        print("Test error is " + "{:.4f}".format(error))
    

if __name__ == '__main__':
    print("PERCEPTRON")
    print("Please choose which part you want to run.")
    print("1 - Standard Perceptron")
    print("2 - Voted Perceptron")
    print("3 - Average Perceptron")
    print("4 - Quit")
    choice = input("Enter your choice: ")

    if choice == '1' or choice == '2' or choice == '3':
        main(int(choice))
