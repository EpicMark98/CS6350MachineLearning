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

            # Add the current item to the dataset
            dataset.append(currItem)
    return dataset

# Gets the prediction
def get_prediction(x, W):
    return np.dot(x, W)

# Calculates the cost
def calc_error(S, W):
    totalError = 0
    for ex in S:
        # Split the example into x and y
        x_i = ex[:-1]
        y_i = ex[-1]

        # Get prediction and calculate error
        pred = get_prediction(x_i, W)
        totalError += (y_i - pred)**2

    return totalError / 2

# Calculates the batch gradient
def calc_gradient_batch(S, W):
    # Initialize variables
    J = [0]*len(W)

    # Go through each example
    for ex in S:
        # Split example into x and y parts
        x_i = ex[:-1]
        y_i = ex[-1]

        # Update the gradient sum
        error = y_i - np.dot(W, x_i)
        for j in range(len(W)):
            J[j] = J[j] + (error * x_i[j])

    J = np.array(J)
    return -1*J

# Calculates the stochastic gradient
def calc_gradient_stochastic(ex, W):
    # Initialize variables
    J = [0]*len(W)

    # Split example into x and y parts
    x_i = ex[:-1]
    y_i = ex[-1]

    # Update the gradient sum
    error = y_i - np.dot(W, x_i)
    for j in range(len(W)):
        J[j] = J[j] + (error * x_i[j])

    J = np.array(J)
    return -1*J

# Runs the stochastic gradient descent algorithm
def gradient_descent_stochastic(train, r, createFig = False):
    # Initialize W
    W = np.array([0]*(len(train[0])-1))
    prevW = np.array([10]*(len(train[0])-1))
    errorThreshold = 0.0001
    errors = []
    error = 1   # Arbitrary starting value

    # Loop until the error stops changing
    while error > errorThreshold:
        prevW = W

        for ex in train:
            # Update the weight vector
            gradient = calc_gradient_stochastic(ex, W)
            W = W - r * gradient

        if createFig:
            # Calculate the total error
            currError = calc_error(train, W)
            errors.append(currError)
        
        # Update error
        error = np.linalg.norm(W - prevW)

    if createFig:
        x = np.arange(1, len(errors)+1)
        plt.plot(x, errors)
        plt.xlabel('Step')
        plt.ylabel('Error')
        plt.title("Training Error")
        plt.savefig("error.png")
        plt.close()

    return W

# Runs the batch gradient descent algorithm
def gradient_descent_batch(train, r, createFig = False):
    # Initialize W
    W = np.array([0]*(len(train[0])-1))
    prevW = np.array([10]*(len(train[0])-1))
    errorThreshold = 0.0001
    errors = []
    error = 1   # Arbitrary starting value

    # Loop until the error stops changing
    while error > errorThreshold:
        prevW = W
        # Update the weight vector
        gradient = calc_gradient_batch(train, W)
        W = W - r * gradient

        # Update error
        error = np.linalg.norm(W - prevW)

        if createFig:
            # Calculate the total error
            currError = calc_error(train, W)
            errors.append(currError)

    if createFig:
        x = np.arange(1, len(errors)+1)
        plt.plot(x, errors)
        plt.xlabel('Step')
        plt.ylabel('Error')
        plt.title("Training Error")
        plt.savefig("error.png")
        plt.close()

    return W

# Wrapper for main
def main(stochastic):
    # Load examples
    trainingData = load_examples()
    testData = load_examples("test.csv")

    # Run gradient descent
    if stochastic:
        finalW = gradient_descent_stochastic(trainingData, 0.05, True)
    else:
        finalW = gradient_descent_batch(trainingData, 0.0125, True)

    # Calculate test error
    error = calc_error(testData, finalW)
    print("Final W: " + str(finalW))
    print("Test error is " + "{:.4f}".format(error))

if __name__ == '__main__':
    print("GRADIENT DESCENT")
    print("Please choose which part you want to run.")
    print("1 - Batch Gradient Descent")
    print("2 - Stochastic Gradient Descent")
    print("3 - Quit")
    choice = input("Enter your choice: ")

    if choice == '1':
        main(False)
    elif choice == '2':
        main(True)
    elif choice == 'o':
        # Calculate the optimal weight vector using the formula from the end of the slides
        train = load_examples()
        X = np.array([row[:-1] for row in train]).T
        Y = np.array([row[-1] for row in train]).T
        temp1 = np.matmul(X, X.T)
        temp2 = np.linalg.inv(temp1)
        temp3 = np.matmul(temp2, X)
        optimal = np.matmul(temp3, Y)
        print(optimal)
