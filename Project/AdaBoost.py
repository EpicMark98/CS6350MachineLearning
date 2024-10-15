from numpy import log, exp, arange, sum
import sys
import matplotlib.pyplot as plt
sys.path.append('../DecisionTree')
import DecisionTree as dt

# Runs through all test examples and calculates the accuracy
def calc_test_accuracy(hypotheses, votes, testData):
    numCorrect = 0
    for example in testData:
        pred = get_prediction(hypotheses, votes, example)
        if pred == example["label"]:
            numCorrect += 1
    return numCorrect / len(testData)

# Gets the AdaBoost prediction for a single example
def get_prediction(h, v, example, average=False):
    # Sanity check
    if len(h) != len(v):
        print("Hypotheses and Votes do not have the same number of elements. This means the programmer screwed up.")
        return "Error"

    # Get the prediction for each weak classifier
    totalPred = 0
    for i in range(len(h)):
        currPred = dt.get_prediction(h[i], example)
        if currPred == "1":   # This condition will need to be expanded to include the positive values for whatever dataset you are using
            totalPred += v[i]
        elif not average:
            totalPred -= v[i]

    # Return the final prediction
    if not average:
        if totalPred >= 0:
            return "1"
        else:
            return "0"
    else:
        return "{:.3f}".format(totalPred/sum(v))

# AdaBoost algorithm
def adaboost(S, attributes, T, testData=None):
    # Initialize return variables
    hypotheses = []
    votes=[]
    trainingError=[]
    testError=[]
    individualTrainingError=[]
    individualTestError=[]

    # Initialize weights of training examples
    for example in S:
        example["weight"] = 1/len(S)

    # Go through the training examples T times
    for t in range(T):
        print("Iteration #" + str(t+1))
        # Find a weak classifier
        node = dt.ID3(S, attributes, splitMetric=dt.SplitMetric.ENTROPY, maxDepth=1)
        hypotheses.append(node)

        # Compute the vote of the classifier
        error = 1-dt.calc_test_accuracy(node, S)
        #individualTrainingError.append(error)
        #individualTestError.append(1-dt.calc_test_accuracy(node, testData))
        vote = 0.5*log((1-error)/error)
        votes.append(vote)

        # Update the weights of the training examples
        Z = 0
        for example in S:
            pred = dt.get_prediction(node, example)
            newWeight = float(example["weight"])
            if pred == example["label"]:
                newWeight = newWeight * exp(-1*vote)
            else:
                newWeight = newWeight * exp(vote)
            Z += newWeight
            example["weight"] = newWeight

        # Normalize the weights
        for example in S:
            weight = float(example["weight"])
            weight = weight / Z
            example["weight"] = weight

        # Report error
        #trainErr = 1-calc_test_accuracy(hypotheses, votes, S)
        #testErr = 1-calc_test_accuracy(hypotheses, votes, testData)
        #trainingError.append(trainErr)
        #testError.append(testErr)
        #print("Training error for T=" + str(t+1) + ": " + "{:.2f}".format(trainErr))
        #print("Test error for T=" + str(t+1) + ": " + "{:.2f}".format(testErr))

    # Create plots
    #x = arange(1,T+1)
    #plt.plot(x, trainingError, label='Training Error')
    #plt.plot(x, testError, '-.', label='Test Error')
    #plt.legend()
    #plt.xlabel('T')
    #plt.ylabel('Error')
    #plt.title("Training and Test Error")
    #plt.savefig("error.png")
    #plt.close()

    #plt.plot(x, individualTrainingError, label='Training Error')
    #plt.plot(x, individualTestError, '-.', label='Test Error')
    #plt.legend()
    #plt.xlabel('T')
    #plt.ylabel('Error')
    #plt.title("Individual Training and Test Error")
    #plt.savefig("individualError.png")
    #plt.close()

    return hypotheses, votes


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

    T = int(input("Enter a T value: "))

    # Run AdaBoost
    hypotheses, votes = adaboost(dataset, attributes, T, testData)

if __name__ == '__main__':
    print("ADABOOST")
    print("Please choose which part you want to run.")
    print("1 - AdaBoost with different T values while calculating training and test error")
    print("2 - Quit")
    choice = input("Enter your choice: ")

    if(choice == '1'):
        main()