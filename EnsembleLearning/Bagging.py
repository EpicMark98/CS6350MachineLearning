import sys
from numpy import arange
import random
import matplotlib.pyplot as plt
sys.path.append('../DecisionTree')
import DecisionTree as dt

# Runs through all test examples and calculates the accuracy
def calc_test_accuracy(hypotheses, testData):
    numCorrect = 0
    for example in testData:
        pred = get_prediction(hypotheses, example)
        if pred == example["label"]:
            numCorrect += 1
    return numCorrect / len(testData)

# Gets the Bagging prediction for a single example
def get_prediction(h, example):
    # Get the prediction for each tree
    netPredYes=0    # Net number of predictions in the "yes" direction (positive means final prediction is "yes")
    for i in range(len(h)):
        currPred = dt.get_prediction(h[i], example)
        if currPred == "yes":   # This condition will need to be expanded to include the positive values for whatever dataset you are using
            netPredYes += 1
        else:
            netPredYes -= 1

    # Return the final prediction
    if netPredYes >= 0:
        return "yes"
    else:
        return "no"

# AdaBoost algorithm
def bagging(S, attributes, T, m, testData, calcError=False):
    # Initialize return variables
    hypotheses = []
    trainingError=[]
    testError=[]

    # Create T trees
    for t in range(T):
        # Randomly sample m examples
        samples=[]
        for i in range(m):
            samples.append(S[random.randint(0,len(S)-1)])

        # Create a decision tree
        node = dt.ID3(samples, attributes, splitMetric=dt.SplitMetric.ENTROPY)
        hypotheses.append(node)

        # Report error
        if calcError:
            trainErr = 1-calc_test_accuracy(hypotheses, S)
            testErr = 1-calc_test_accuracy(hypotheses, testData)
            trainingError.append(trainErr)
            testError.append(testErr)
            print("Training error for T=" + str(t) + ": " + "{:.2f}".format(trainErr))
            print("Test error for T=" + str(t) + ": " + "{:.2f}".format(testErr))

    # Create plots
    if calcError:
        x = arange(1,T+1)
        plt.plot(x, trainingError, label='Training Error')
        plt.plot(x, testError, '-.', label='Test Error')
        plt.legend()
        plt.xlabel('T')
        plt.ylabel('Error')
        plt.title("Training and Test Error")
        plt.savefig("error.png")
        plt.close()

    return hypotheses


# Wrapper for main so that variables are not global
def main():
    # Create the attribute set by parsing the data-desc.txt file
    attributes, attributeNameList = dt.create_attribute_set();

    # Load the training examples
    dataset, isUnknown = dt.load_examples(attributeNameList)

    # Convert numeric attributes to binary ones
    dt.convert_numeric_to_binary(dataset, attributes)

    # Load test data
    testData, isUnknown = dt.load_examples(attributeNameList, "test.csv")

    # Convert numeric attributes to binary ones
    dt.convert_numeric_to_binary(testData, attributes)

    # Run AdaBoost
    hypotheses = bagging(dataset, attributes, 80, 3000, testData)

if __name__ == '__main__':
    main()