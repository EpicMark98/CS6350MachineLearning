import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

# Gets the prediction for linear SVM
def get_prediction(x, W):
    return 1 if np.dot(x, W) >= 0 else -1

# Calculates the cost for linear SVM
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

X = None
Y = None
K = None
gamma = 1

# Define a kernel function
def kernel(x1, x2):
    return np.exp(-1*(np.linalg.norm(x1-x2) ** 2) / gamma)

# Define SVM dual objective
def objective(alpha):
    term1 = 0.5 * np.sum(np.outer(alpha, alpha) * np.outer(Y, Y) * K)
    term2 = np.sum(alpha)
    return term1 - term2


# Equality constraints
def eq_constraint(alpha):
    return np.dot(alpha, Y)     # Sum of alpha_i * y_i = 0

def callback_fun(arg):
    print("Iteration finished")

# Calculates the cost for nonlinear SVM
def calc_error_nonlin(x_train, y_train, x_test, y_test, alphas, B):
    totalError = 0
    predictions = predict(x_train, y_train, alphas, x_test, B)
    for i in range(len(predictions)):
        if y_test[i] != predictions[i]:
            totalError += 1

    return totalError / len(x_test)

# Prediction function
def predict(X_train, y_train, alphas, X_test, B):
    # Identify support vectors (non-zero alphas)
    support_vector_indices = np.where(alphas > 1e-5)[0]
    support_vectors = X_train[support_vector_indices]
    support_alphas = alphas[support_vector_indices]
    support_labels = y_train[support_vector_indices]
    
    # Decision function for test data
    def decision_function(x):
        kernel_values = np.exp(
            -gamma * np.sum((support_vectors - x)**2, axis=1)
        )
        return np.sum(support_alphas * support_labels * kernel_values) + B
    
    # Compute predictions
    predictions = np.array([np.sign(decision_function(x)) for x in X_test])
    return predictions

def calc_WT_times_transformed(x, y, x_test, alphas):
    return sum([alphas[i] * y[i] * kernel(x[i], x_test) for i in range(len(x))])

# Run the dual optimization problem
def dual():
    global X
    global Y
    global K

    # Load examples
    trainingData = np.array(load_examples(foldB = False))
    X = trainingData[:,:-1]
    Y = trainingData[:, -1]
    c_values = [100/873, 500/873, 700/873]

    # Compute Gram matrix for linear kernel
    K = np.dot(X, X.T)

    print("PART A: Linear SVM")
    for C in c_values:
        print("Running dual optimization for C = " + str(C))

        # Initialize alphas
        alpha0 = np.zeros(len(X))

        # Intialize inequality constraints
        bounds = [(0, C) for i in range(len(X))]

        # Optimize
        result = minimize(objective, alpha0,  constraints={'type':'eq', 'fun': eq_constraint}, bounds = bounds, callback=callback_fun)

        # Recover W and B
        alphas = result.x
        W = sum(alphas[i] * Y[i] * X[i] for i in range(len(X)))
        b = 0
        for i in range(len(X)):
            b += Y[i] - np.dot(W, X[i])
        b /= len(X)
        W = np.insert(W, 0, b)
        print(W)

        # Load data with B folded in
        trainingData = load_examples()
        testData = load_examples("test.csv")
        training_error = calc_error(trainingData, W)
        test_error = calc_error(testData, W)
        print("Training error: " + str(training_error))
        print("Test error: " + str(test_error))

    print("PART B: Nonlinear SVM")

    # Compute Gram matrix for nonlinear kernel
    K = np.array([[kernel(xi, xj) for xj in X] for xi in X])

    for C in c_values:
        print("Running dual optimization for C = " + str(C))

        # Initialize alphas
        alpha0 = np.zeros(len(X))

        # Intialize inequality constraints
        bounds = [(0, C) for i in range(len(X))]

        # Optimize
        result = minimize(objective, alpha0,  constraints={'type':'eq', 'fun': eq_constraint}, bounds = bounds, callback=callback_fun)
        alphas = result.x

        # Compute B
        B = 0
        for i in range(len(alphas)):
            B += Y[i] - calc_WT_times_transformed(X, Y, i, alphas)
        B /= len(alphas)

        # Print support vectors
        sup_vec = []
        for i in range(len(alphas)):
            if alphas[i] > 1E-6:
                sup_vec.append(X[i])
        print("Num support vectors: " + str(len(sup_vec)))
        print(sup_vec)

        # Load test data
        testData = np.array(load_examples("test.csv", False))
        test_X = testData[:,:-1]
        test_Y = testData[:, -1]
        training_error = calc_error_nonlin(X, Y, X, Y, alphas, B)
        test_error = calc_error_nonlin(X, Y, test_X, test_Y, alphas, B)
        print("Training error: " + str(training_error))
        print("Test error: " + str(test_error))
        
    

if __name__ == '__main__':
    print("SVM")
    print("Please choose which part you want to run.")
    print("1 - SGD")
    print("2 - Dual")
    print("3 - Quit")
    choice = input("Enter your choice: ")

    if choice == '1':
        main()
    elif choice == '2':
        dual()
