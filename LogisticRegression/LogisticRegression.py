import numpy as np
import matplotlib.pyplot as plt

# Loads the data into a list of numpy arrays.
def load_examples(filename = "train.csv", foldB = True):
    dataset = []
    with open(filename, 'r') as f:
        # For each example
        for line in f:
            # Get the attribute values and add them to a numpy array
            stringArray = line.strip().split(',')
            currItem = None
            if foldB:
                currItem = np.array([1] + [float(x) for x in stringArray])  # Convert strings and add 1 for bias
            else:
                currItem = np.array([float(x) for x in stringArray])  # Convert strings

            # Replace 0 labels with -1
            if currItem[-1] == 0:
                currItem[-1] = -1

            # Add the current item to the dataset
            dataset.append(currItem)
    return dataset

# Gets the prediction
def get_prediction(x, W):
    return 1 if np.dot(x, W) >= 0 else -1

# Calculates the cost
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

# Computes the stochastic gradient of the loss. If sigma is nonzero, it uses MAP estimation. Otherwise, it uses ML
def calc_gradient(x_i, y_i, W, numEx, sigma):
    wtx = np.dot(x_i, W)
    exp = np.exp(-1 * y_i * wtx)
    numerator = -1 * y_i * x_i * exp
    result = (numerator / (1 + exp)) * numEx

    if sigma != 0:
        result += (2 / (sigma**2)) * W

    return result

# Runs the stochastic gradient descent for logistic regession MAP
def map_sgd(train, gamma_0, sigma, epochs, test):
    # Initialize W
    W = np.array([0]*(len(train[0])-1))
    training_error = []
    test_error = []
    alpha = 0.1

    # Loop until the error stops changing
    for e in range(epochs):
        # Update gamma according to the schedule
        gamma = gamma_0 / (1 + (gamma_0/alpha) * e)

        np.random.shuffle(train)
        for ex in train:
            x_i = ex[:-1]
            y_i = ex[-1]

            # Update the weights
            W = W - gamma * calc_gradient(x_i, y_i, W, len(train), sigma)

        # Compute training and test error
        training_error.append(calc_error(train, W)) 
        test_error.append(calc_error(test, W)) 

        if (e+1) % 100 == 0:
            print("EPOCH " + str(e+1) + "/" + str(epochs))

    print("Final training error: " + str(training_error[-1]))
    print("Final test error: " + str(test_error[-1]))

    return W

# Runs the stochastic gradient descent for logistic regession MLE
def mle_sgd(train, gamma_0, epochs, test):
    # Initialize W
    W = np.array([0]*(len(train[0])-1))
    training_error = []
    test_error = []
    alpha = 0.1

    # Loop until the error stops changing
    for e in range(epochs):
        # Update gamma according to the schedule
        gamma = gamma_0 / (1 + (gamma_0/alpha) * e)

        np.random.shuffle(train)
        for ex in train:
            x_i = ex[:-1]
            y_i = ex[-1]

            # Update the weights
            W = W - gamma * calc_gradient(x_i, y_i, W, len(train), 0)

        # Compute training and test error
        training_error.append(calc_error(train, W)) 
        test_error.append(calc_error(test, W)) 

        if (e+1) % 100 == 0:
            print("EPOCH " + str(e+1) + "/" + str(epochs))

    print("Final training error: " + str(training_error[-1]))
    print("Final test error: " + str(test_error[-1]))

    return W

# Wrapper for main
def main(choice):
    # Load examples
    trainingData = load_examples()
    testData = load_examples("test.csv")
    sigmas = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

    if choice:
        for sigma in sigmas:
            print("Running MAP SGD with sigma " + str(sigma))
            finalW = map_sgd(trainingData, 0.00000001, sigma, 500, testData)
            print("Final W: " + str(finalW))
    else:
        print("Running MLE SGD")
        finalW = mle_sgd(trainingData, 0.000001, 500, testData)
        print("Final W: " + str(finalW))

if __name__ == '__main__':
    print("Logistic Regression")
    print("Please choose which part you want to run.")
    print("1 - MAP")
    print("2 - MLE")
    print("3 - Quit")
    choice = input("Enter your choice: ")

    if choice == '1' or choice == '2':
        main(choice == '1')

