from os import replace
import numpy as np
import statistics as st

# Helper method to parse the data-desc.txt file. Returns a dictionary of attribute name as key and lsit of attribute values as value
def create_attribute_set():
    attributes = {}
    attributeNameList = [] # Ordered list of attribute names needed only for loading the dataset
    with open("data-desc.txt", 'r') as f:
        for line in f:
            sections = line.strip().split(':')

            # Skip lines that don't have an attribute
            if len(sections) == 1:
                continue

            # Add the list of attribute values to the dictionary
            attributeValues = sections[1].strip('.').strip().split(',')

            # Remove whitespace
            for i in range(len(attributeValues)):
                attributeValues[i] = attributeValues[i].strip()

            attributes[sections[0]] = attributeValues
            attributeNameList.append(sections[0])
    return attributes, attributeNameList

# Loads the data into a list of dictionaries. Also returns a boolean indicating if any values were "unknown"
def load_examples(attributes, filename = "train.csv"):
    dataset = []
    foundUnknown = False
    with open(filename, 'r') as f:
        # For each example
        for line in f:
            # Get the attribute values
            items = line.strip().split(',')
            currItem = {}

            # For each attribute value, add it to a dictionary with the attribute name
            for i in range(len(items) - 1):
                if items[i].lower() == "unknown":
                    foundUnknown = True
                currItem[attributes[i]] = items[i]

            # Add the label also
            currItem["label"] = items[len(items) - 1]

            # Add the current item to the dataset
            dataset.append(currItem)
    return dataset, foundUnknown

# Check if the dataset contains numeric attibute values and if so, replace it with a binary thresholding.
def convert_numeric_to_binary(dataset, attributes):
    if len(dataset) == 0:
        return

    # Check first item for numeric attribute values and put them in a list
    numericAttributes = []
    for key in dataset[0].keys():
        if dataset[0][key].strip("-").strip(".").isnumeric():
            numericAttributes.append(key)

    # For each numeric attribute
    for key in numericAttributes:
        median = 0
        # Check if the median was already calculated. If so, use it
        if len(attributes[key]) == 2 and attributes[key][0].find("<") != -1:
            median = float(attributes[key][0].strip("<"))
        else:
            # Calculate the median value
            numbers = []
            try:
                for item in dataset:
                    numbers.append(float(item[key]))
            except:
                # Non-numeric value found. Don't process this key anymore
                continue
            median = st.median(numbers)
            # Set the new correct attribute values
            attributes[key] = ["<" + str(median), ">=" + str(median)]

        # Check if each value is greater or less than the median and assign one of two strings
        for item in dataset:
            if float(item[key]) < median:
                item[key] = "<" + str(median)
            else:
                item[key] = ">=" + str(median)

# Replaces any "unknown" values with the most common value
def replace_unknown_values(dataset, attributes):
    # Calculate all most common values
    mostCommonValues = {}
    for a in attributes.keys():
        mostCommonValues[a] = find_most_common_value(dataset, a)

    # Replace any unknowns
    for item in dataset:
        for key in item.keys():
            if item[key].lower() == "unknown":
                item[key] = mostCommonValues[key]

# Enum to choose which metric to use when splitting the data
class SplitMetric():
    ENTROPY = 1
    ME = 2
    GINI = 3

# Class to hold a decision tree node
class DecisionTreeNode:
    def __init__(self):
        self.isLeaf = False  # Marks if the node is a leaf or not
        self.childNodes = {} # Dictionary of attribute value (string) to DecisionTreeNode                   
        self.value = ""      # If leaf, contains label. Otherwise, contains attribute name that is being split on

# Returns true if all items in S have the same label and false otherwise
def has_unique_labels(S):
    if len(S) == 0:
        return True
    label = S[0]["label"]
    for item in S:
        if item["label"] != label:
            return False
    return True

# Returns a dictionary of counts for each possible attribute value
def get_counts(S, attribute):
    counts = {}
    for item in S:
        value = item[attribute]
        if value in counts.keys():
            counts[value] += 1
        else:
            counts[value] = 1

    return counts

# Returns the most common value in S given attribute str. Default is for labels
def find_most_common_value(S, str = "label"):
    # Count the labels first
    labelCounts = get_counts(S, str)

    # Find and return the label with the most
    countMax = 0
    labelMax = None
    for key in labelCounts.keys():
        if labelCounts[key] > countMax:
            countMax = labelCounts[key]
            labelMax = key
    return labelMax

# Creates a subset of S where the items have the given attribute value
def get_subset(S, attributeName, attributeValue):
    subset = []
    for item in S:
        if item[attributeName] == attributeValue:
            subset.append(item)
    return subset

# Calculates the entropy of S. If attribute is provided, then calculates entropy of S given for each possible value of attribute
def calc_entropy(S, attribute = None):
    if len(S) == 0:
        return 0

    # Calculate entropy of S
    if attribute == None:
        labelCounts = get_counts(S, "label")
        entropy = 0
        for count in labelCounts.values():
            prob = count / len(S)
            if prob != 0:
                entropy -= prob * np.log2(prob)
        return entropy

    counts = get_counts(S, attribute)
    entropy = 0
    # Calculate the entropy for each attribute value
    for attributeValue in counts.keys():
        subset = get_subset(S, attribute, attributeValue)
        currEntropy = calc_entropy(subset)      # Entropy of this attribute value
        prob = counts[attributeValue] / len(S)  # Probability of getting this attribute value
        entropy += prob * currEntropy           # Calculate average
    return entropy

# Calculates the majority error of S. If attribute is provided, then calculates majority error of S given for each possible value of attribute
def calc_me(S, attribute = None):
    if len(S) == 0:
        return 0

    # Calculate majority error of S
    if attribute == None:
        labelCounts = get_counts(S, "label")
        maxCount = max(labelCounts.values())
        return 1 - maxCount / len(S)

    counts = get_counts(S, attribute)
    me = 0
    # Calculate the majority error for each attribute value
    for attributeValue in counts.keys():
        subset = get_subset(S, attribute, attributeValue)
        currME = calc_me(subset)              # Majority error of this attribute value
        prob = counts[attributeValue] / len(S)  # Probability of getting this attribute value
        me += prob * currME                     # Calculate average
    return me

# Calculates the gini index of S. If attribute is provided, then calculates gini index of S given for each possible value of attribute
def calc_gini(S, attribute = None):
    if len(S) == 0:
        return 0

    # Calculate gini index of S
    if attribute == None:
        labelCounts = get_counts(S, "label")
        gini = 0
        for count in labelCounts.values():
            prob = count / len(S)
            gini += prob**2
        return 1 - gini

    counts = get_counts(S, attribute)
    gini = 0
    # Calculate the entropy for each attribute value
    for attributeValue in counts.keys():
        subset = get_subset(S, attribute, attributeValue)
        currGini = calc_gini(subset)            # Gini index of this attribute value
        prob = counts[attributeValue] / len(S)  # Probability of getting this attribute value
        gini += prob * currGini                  # Calculate average
    return gini

# Calculates the information gain for each attribute and returns the best one
def get_best_attribute(S, attributes, splitMetric = SplitMetric.ENTROPY):
    # Go through all the attributes and calculate the gain
    bestAttribute = None
    maxGain = -1  # Gain should alwasy be non-negative but this default ensures that an attribute is chosen if all gains are 0
    for a in attributes:
        gain = 0
        if splitMetric == SplitMetric.ENTROPY:
            gain = calc_entropy(S) - calc_entropy(S, a)
        elif splitMetric == SplitMetric.ME:
            gain = calc_me(S) - calc_me(S, a)
        elif splitMetric == SplitMetric.GINI:
            gain = calc_gini(S) - calc_gini(S, a)
        if gain > maxGain:
            maxGain = gain
            bestAttribute = a

    return bestAttribute

# ID3 learning algorithm
# S is current data subset
# attributes is current attribute list
# splitMetric is the choice of metric to use when determining which attribute to split the data on
def ID3(S, attributes, splitMetric = SplitMetric.ENTROPY, maxDepth = 100000, currDepth = 1):
    # Handle base cases of unique labels and empty attribute set
    if has_unique_labels(S) or len(attributes.keys()) == 0 or currDepth > maxDepth:
        leafNode = DecisionTreeNode()
        leafNode.isLeaf = True
        leafNode.value = find_most_common_value(S)
        return leafNode

    # Create a root node for the tree
    rootNode = DecisionTreeNode()

    # Choose the attribute that best splits S
    a = get_best_attribute(S, attributes.keys(), splitMetric)
    rootNode.value = a

    for value in attributes[a]:
        # Add a new tree branch
        subset = get_subset(S, a, value)
        if len(subset) == 0:
            rootNode.childNodes[value] = DecisionTreeNode()
            rootNode.childNodes[value].isLeaf = True
            rootNode.childNodes[value].value = find_most_common_value(S)
        else:
            # Remove a from attributes
            attributeSubset = attributes.copy()
            attributeSubset.pop(a)

            # Recursive call
            rootNode.childNodes[value] = ID3(subset, attributeSubset, splitMetric, maxDepth, currDepth + 1)

    return rootNode

# Traverses the tree to get and return a prediction
def get_prediction(rootNode, testExample):
    currNode = rootNode

    # End of tree has been reached so return label
    if currNode.isLeaf:
        return currNode.value

    key = currNode.value        # Which attribute are we splitting on?
    value = testExample[key]    # What value does the example have for this attribute
    return get_prediction(rootNode.childNodes[value], testExample)    # Travel to the cooresponding node and get the prediction

# Runs through all test examples and calculates the accuracy
def calc_test_accuracy(rootNode, testData):
    numCorrect = 0
    for example in testData:
        pred = get_prediction(rootNode, example)
        if pred == example["label"]:
            numCorrect += 1
    return numCorrect / len(testData)

# Wrapper so that variables are actually local
def main():
    # Create the attribute set by parsing the data-desc.txt file
    attributes, attributeNameList = create_attribute_set();

    # Load the training examples
    dataset, isUnknown = load_examples(attributeNameList)

    # Convert numeric attributes to binary ones
    convert_numeric_to_binary(dataset, attributes)

    # Get user settings
    metricStr = input("Which metric would you like to use for splitting? (E - entropy, M - majority error, G - gini index)").upper()
    depth = int(input("Please enter a maximum tree depth: " ))
    replaceUnknown = False

    # If unknown entries were found, ask if the user wants to replace them
    if isUnknown:
        replaceUnknown = input("Would you like to replace unknown attribute values with the most common? (Y/N) " ).upper() == "Y"

    metric = SplitMetric.ENTROPY
    if(metricStr == "M"):
        print("Running ID3 with Majority Error to a max depth of " + str(depth))
        metric = SplitMetric.ME
    elif(metricStr == "G"):
        print("Running ID3 with Gini Index to a max depth of " + str(depth))
        metric = SplitMetric.GINI
    else:
        print("Running ID3 with Entropy to a max depth of " + str(depth))

    # Replace unknown values
    if replaceUnknown:
        replace_unknown_values(dataset, attributes)
    
    # Run ID3
    rootNode = ID3(dataset, attributes, metric, depth)

    # Load test data
    testData, isUnknown = load_examples(attributeNameList, "test.csv")

    # Convert numeric attributes to binary ones
    convert_numeric_to_binary(testData, attributes)

    # Replace unknown values
    if replaceUnknown:
        replace_unknown_values(testData, attributes)

    # Print the results
    print("Training error: " + str(1-calc_test_accuracy(rootNode, dataset)))
    print("Test error: " + str(1-calc_test_accuracy(rootNode, testData)))

if __name__ == '__main__':
    main()
