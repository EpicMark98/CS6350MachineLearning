import numpy as np
import matplotlib.pyplot as plt

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

# Calculates the RMSE
def calc_error(S, W):
    totalError = 0
    for ex in S:
        # Split the example into x and y
        x_i = ex[:-1]
        y_i = ex[-1]

        # Get prediction and calculate error
        pred = get_prediction(x_i, W)
        totalError += (y_i - pred)**2

    return totalError / (2 * len(S))

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

# Runs the gradient descent algorithm
def gradient_descent(train, stochastic, r, createFig = False):
    # Initialize W
    W = np.array([0]*(len(train[0])-1))
    prevW = np.array([10]*(len(train[0])-1))
    errorThreshold = 0.0001
    errors = []

    # Loop until the error stops changing
    while np.linalg.norm(W - prevW) > errorThreshold:
        prevW = W
        # Update the weight vector
        if stochastic:
            print("Not implemented")
            return None
        else:
            gradient = calc_gradient_batch(train, W)
            W = W - r * gradient

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
    finalW = gradient_descent(trainingData, stochastic, 0.0125, True)

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