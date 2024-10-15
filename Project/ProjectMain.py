import DecisionTree as dt

def RunDecisionTree(S, A, names, test):
    print("Running decision tree algorithm")

    # Convert numeric attributes to binary ones
    dt.convert_numeric_to_binary(S, A)
    dt.convert_numeric_to_binary(test, A)

    # Run ID3
    rootNode = dt.ID3(S, A, dt.SplitMetric.GINI, 20)

    # Generate predictions
    with open("predictions.csv", 'w+') as f:
        f.write("ID,Prediction\n")
        for ex in test:
            pred = dt.get_prediction(rootNode, ex)
            f.write(ex["id"] + "," + pred + "\n")

def main():
    # Load the data
    print("Loading the data, please wait...")
    attributes, attributeNames = dt.create_attribute_set()
    trainingData, foundUnknown = dt.load_examples(attributeNames)

    # Replace unknown values
    dt.replace_unknown_values(trainingData, attributes)     # TODO: change me to use averages when attribute is numeric

    attributeNames.insert(0, 'id')  # Add ID because it is used in test data

    testData, foundUnknown = dt.load_examples(attributeNames, 'test.csv')

    # Replace unknown values
    dt.replace_unknown_values(testData, attributes)     # TODO: change me to use averages when attribute is numeric

    # Get user choice
    print("Choose your algorithm")
    print("1 - Decision Tree")
    choice = input("Enter your choice: ")

    if choice == '1':
        RunDecisionTree(trainingData, attributes, attributeNames, testData)

if __name__ == '__main__':
    main()