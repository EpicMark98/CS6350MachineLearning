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

# Runs the stochastic gradient descent for SMV
def svm_sgd(train, gamma_0, C, schedule, epochs, test):
    # Initialize W
    W = np.array([0]*(len(train[0])-1))
    training_error = []
    test_error = []
    alpha = 0.000008

    # Loop until the error stops changing
    for e in range(epochs):

        # Update gamma according to the schedule
        if schedule == 1:
            gamma = gamma_0 / (1 + (gamma_0/alpha) * e)
        else:
            gamma = gamma_0 / (1 + e)

        np.random.shuffle(train)
        for ex in train:
            x_i = ex[:-1]
            y_i = ex[-1]
            w_0 = np.insert(W[1:], 0, 0)

            # Update the weights
            val = y_i * np.dot(x_i, W)
            if val <= 1:
                W = W - gamma * w_0 + gamma * C * len(train) * y_i * x_i
            else:
                W = np.insert(w_0[1:] * (1-gamma), 0, W[0])

        # Compute training and test error
        training_error.append(calc_error(train, W)) 
        test_error.append(calc_error(test, W)) 

    # Create plots
    x = np.arange(1,epochs+1)
    plt.plot(x, training_error, label='Training Error')
    plt.plot(x, test_error, '-.', label='Test Error')
    plt.legend()
    plt.xlabel('T')
    plt.ylabel('Error')
    plt.title("Training and Test Error")
    plt.savefig("error_C" + str(C) + "_sch" + str(schedule) + ".png")
    plt.close()

    print("Final training error: " + str(training_error[-1]))
    print("Final test error: " + str(test_error[-1]))

    return W

# Wrapper for main
def main():
    # Load examples
    trainingData = load_examples()
    testData = load_examples("test.csv")
    c_values = [100/873, 500/873, 700/873]

    # Run SVM
    for c in c_values:
        print("Running SVM SGD schedule A with C value " + str(c))
        finalW = svm_sgd(trainingData, 0.00001, c, 1, 100, testData)
        print("Final W: " + str(finalW))

        print("Running SVM SGD schedule B with C value " + str(c))
        finalW = svm_sgd(trainingData, 0.000005, c, 2, 100, testData)
        print("Final W: " + str(finalW))

if __name__ == '__main__':
    print("SVM")
    print("Please choose which part you want to run.")
    print("1 - SGD")
    print("2 - Quit")
    choice = input("Enter your choice: ")

    if choice == '1':
        main()
