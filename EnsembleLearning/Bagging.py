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

    T = input("Enter a T value: ")

    # Run AdaBoost
    hypotheses = bagging(dataset, attributes, T, 3000, testData, True)

def bias_and_variance():

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

    # Initialize variables
    baggedPredictors = []

    # Create predictors (30 bags each with 70 trees)
    for i in range(30):
        print("Creating bag #" + str(i+1))

        # Randomly sample 1000 examples
        samples=[]
        for i in range(1000):
            samples.append(dataset[random.randint(0,len(dataset)-1)])

        # Run AdaBoost
        hypotheses = bagging(dataset, attributes,70, 500, testData)

        # Add this bagged predictor to the list
        baggedPredictors.append(hypotheses)

    # Calculate bias and variance on single predictors
    print("Calculating stats for single trees")
    averageBias = 0
    averageVariance = 0
    for ex in testData:
        averagePrediction = 0
        averageTruth = 0
        varianceSum = 0
        for bag in baggedPredictors:
            singleTree = bag[0]
            pred = dt.get_prediction(singleTree, ex)

            # Add the predictions into the running sum for bias calculation
            if pred == "yes":
                averagePrediction += 1
            else:
                averagePrediction -= 1
            if ex["label"] == "yes":
                averageTruth += 1
            else:
                averageTruth -= 1

            # Add the current variance into the sum
            if pred != ex["label"]:
                varianceSum += 4    # When the labels are different, the variance is 4. When they are the same, the variance is 0

        # Average the predictions and truth
        averagePrediction /= len(baggedPredictors)
        averageTruth /= len(baggedPredictors)

        # Calculate bias
        bias = (averagePrediction - averageTruth) ** 2
        averageBias += bias

        # Calculate variance
        variance = varianceSum / (len(baggedPredictors)-1)
        averageVariance += variance

    # Average the bias and variance
    averageBias /= len(testData)
    averageVariance /= len(testData)

    # Print the results
    print("Average bias for single trees: " + "{:.4f}".format(averageBias))
    print("Average variance for single trees: " + "{:.4f}".format(averageVariance))
    print("Estimated squared error: " + "{:.4f}".format(averageBias + averageVariance))

    # Calculate bias and variance on bagged predictors\
    print("Calculating stats for bagged trees")
    averageBias = 0
    averageVariance = 0
    for ex in testData:
        averagePrediction = 0
        averageTruth = 0
        varianceSum = 0
        for bag in baggedPredictors:
            pred = get_prediction(bag, ex)

            # Add the predictions into the running sum for bias calculation
            if pred == "yes":
                averagePrediction += 1
            else:
                averagePrediction -= 1
            if ex["label"] == "yes":
                averageTruth += 1
            else:
                averageTruth -= 1

            # Add the current variance into the sum
            if pred != ex["label"]:
                varianceSum += 4    # When the labels are different, the variance is 4. When they are the same, the variance is 0

        # Average the predictions and truth
        averagePrediction /= len(baggedPredictors)
        averageTruth /= len(baggedPredictors)

        # Calculate bias
        bias = (averagePrediction - averageTruth) ** 2
        averageBias += bias

        # Calculate variance
        variance = varianceSum / (len(baggedPredictors)-1)
        averageVariance += variance

    # Average the bias and variance
    averageBias /= len(testData)
    averageVariance /= len(testData)

    # Print the results
    print("Average bias for bagged trees: " + "{:.4f}".format(averageBias))
    print("Average variance for bagged trees: " + "{:.4f}".format(averageVariance))
    print("Estimated squared error: " + "{:.4f}".format(averageBias + averageVariance))

if __name__ == '__main__':
    print("BAGGING")
    print("Please choose which part you want to run.")
    print("1 - Bagging with different T values while calculating training and test error")
    print("2 - Bias and Variance Calculation")
    print("3 - Quit")
    choice = input("Enter your choice: ")

    if(choice == '1'):
        main()
    elif(choice == '2'):
        bias_and_variance()