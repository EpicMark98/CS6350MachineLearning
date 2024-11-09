import DecisionTree as dt
import Bagging as bg
import AdaBoost as ab
import RandomForests as rf
import Perceptron as pc

# Runs the decision tree algorithm
def RunDecisionTree(S, A, test):
    depth = int(input("Enter a depth to limit to: "))

    print("Running decision tree algorithm")

    # Convert numeric attributes to binary ones
    dt.convert_numeric_to_binary(S, A)
    dt.convert_numeric_to_binary(test, A)

    # Run ID3
    rootNode = dt.ID3(S, A, dt.SplitMetric.GINI, depth)

    # Generate predictions
    with open("predictions.csv", 'w+') as f:
        f.write("ID,Prediction\n")
        for ex in test:
            pred = dt.get_prediction(rootNode, ex)
            f.write(ex["id"] + "," + pred + "\n")

# Runs the bagging algorithm
def RunBagging(S, A, test):
    depth = int(input("Enter a depth to limit to: "))

    # Convert numeric attributes to binary ones
    dt.convert_numeric_to_binary(S, A)
    dt.convert_numeric_to_binary(test, A)

    T = int(input("Enter a T value: "))

    print("Running bagging algorithm")

    # Run Bagging
    hypotheses = bg.bagging(S, A, T, 2000, None, False)

    # Generate predictions
    with open("predictions.csv", 'w+') as f:
        f.write("ID,Prediction\n")
        for ex in test:
            pred = bg.get_prediction(hypotheses, ex, True)
            f.write(ex["id"] + "," + pred + "\n")

# Runs the random forest algorithm
def RunRandomForest(S, A, test):
    depth = int(input("Enter a depth to limit to: "))

    # Convert numeric attributes to binary ones
    dt.convert_numeric_to_binary(S, A)
    dt.convert_numeric_to_binary(test, A)

    T = int(input("Enter a T value: "))
    n = int(input("Enter a attribute subset size: "))

    print("Running Random Forest algorithm")

    # Run Random Forest
    hypotheses = rf.random_forest(S, A, T, 2000, None, n, False)

    # Generate predictions
    with open("predictions.csv", 'w+') as f:
        f.write("ID,Prediction\n")
        for ex in test:
            pred = rf.get_prediction(hypotheses, ex, True)
            f.write(ex["id"] + "," + pred + "\n")

# Runs the AdaBoost algorithm
def RunAdaBoost(S, A, test):
    print("Running AdaBoost")
    # Convert numeric attributes to binary ones
    dt.convert_numeric_to_binary(S, A)
    dt.convert_numeric_to_binary(test, A)

    T = int(input("Enter a T value: "))

    # Run AdaBoost
    hypotheses, votes = ab.adaboost(S, A, T, None)

    # Generate predictions
    with open("predictions.csv", 'w+') as f:
        f.write("ID,Prediction\n")
        for ex in test:
            pred = ab.get_prediction(hypotheses, votes, ex, True)
            f.write(ex["id"] + "," + pred + "\n")

# Runs the Perceptron algorithm
def RunPerceptron(S, test):
    print("Running Perceptron")
    final_W = pc.average_perceptron(S, 0.2, 10)

    # Generate predictions
    print("Generating predictions")
    with open("predictions.csv", 'w+') as f:
        f.write("ID,Prediction\n")
        for ex in test:
            pred = pc.get_prediction(ex[1:], final_W)
            f.write(str(int(ex[0])) + "," + str(pred) + "\n")


def main():
    # Get user choice
    print("Choose your algorithm")
    print("1 - Decision Tree")
    print("2 - Bagging")
    print("3 - Random Forest")
    print("4 - AdaBoost")
    print("5 - Averaged Perceptron")
    choice = input("Enter your choice: ")

    # Load the data
    print("Loading the data, please wait...")
    if choice == '5':
        attributes, attributeNames = dt.create_attribute_set()
        trainingData = pc.load_examples(attributes, attributeNames)
        testData = pc.load_examples(attributes, attributeNames, "test.csv")
    else:
        attributes, attributeNames = dt.create_attribute_set()
        trainingData, foundUnknown = dt.load_examples(attributeNames)

        # Replace unknown values
        dt.replace_unknown_values(trainingData, attributes)     # TODO: change me to use averages when attribute is numeric

        attributeNames.insert(0, 'id')  # Add ID because it is used in test data

        testData, foundUnknown = dt.load_examples(attributeNames, 'test.csv')

        # Replace unknown values
        dt.replace_unknown_values(testData, attributes)     # TODO: change me to use averages when attribute is numeric

    if choice == '1':
        RunDecisionTree(trainingData, attributes, testData)
    elif choice == '2':
        RunBagging(trainingData, attributes, testData)
    elif choice == '3':
        RunRandomForest(trainingData, attributes, testData)
    elif choice == '4':
        RunAdaBoost(trainingData, attributes, testData)
    elif choice == '5':
        RunPerceptron(trainingData, testData)

if __name__ == '__main__':
    main()