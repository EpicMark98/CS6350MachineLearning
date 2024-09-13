import numpy as np

# Helper method to parse the data-desc.txt file. Returns an ordered list of strings representing the attributes
def create_attribute_set():
    # Use a boolean to flag when we have reached the column labels section
    foundAttributes = False
    attributes = []
    with open("data-desc.txt", 'r') as f:
        for line in f:
            if foundAttributes:                    # This line has the attributes so split the line and ignore the 'labels' entry
                for item in line.strip().split(','):
                    if (item != "label"):
                        attributes.append(item)
            elif line.startswith("| columns"):     # The next line is the attributes so set the flag now
                foundAttributes = True
    return attributes

# Loads the training data into a list of dictionaries from train.csv
def load_training_examples():
    dataset = []
    with open("train.csv", 'r') as f:
        # For each example
        for line in f:
            # Get the attribute values
            items = line.strip().split(',')
            currItem = {}

            # For each attribute value, add it to a dictionary with the attribute name
            for i in range(len(items) - 1):
                currItem[attributes[i]] = items[i]

            # Add the label also
            currItem["label"] = items[len(items) - 1]

            # Add the current item to the dataset
            dataset.append(currItem)
    return dataset

# Enum to choose which metric to use when splitting the data
class SplitMetric():
    ENTROPY = 1
    ME = 2
    GINI = 3

# Class to hold a decision tree node
class DecisionTreeNode:
    isLeaf = False  # Marks if the node is a leaf or not
    childNodes = {} # Dictionary of attribute value (string) to DecisionTreeNode                   
    value = ""      # If leaf, contains label. Otherwise, contains attribute name that is being split on

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

# Returns the most common label in S
def find_most_common_label(S):
    # Count the labels first
    labelCounts = get_counts(S, "label")

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
        for count in labelCounts.values:
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
        currME = calc_gini(subset)              # Majority error of this attribute value
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
        for count in labelCounts.values:
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
    return 1 - gini

# Calculates the information gain for each attribute and returns the bets one
def get_best_attribute(S, attributes, splitMetric = SplitMetric.ENTROPY):
    values = [] # List of entropy/me/gini values cooresponding to each attribute

    # Go through all the attributes
    for a in attributes:



# ID3 learning algorithm
# S is current data subset
# attributes is current attribute list
# splitMetric is the choice of metric to use when determining which attribute to split the data on
def ID3(S, attributes, splitMetric = SplitMetric.ENTROPY):
    # Handle base cases of unique labels and empty attribute set
    if has_unique_labels(S) or len(attributes) == 0:
        leafNode = DecisionTreeNode()
        leafNode.isLeaf = True
        leafNode.value = find_most_common_label(S)
        return leafNode


if __name__ == '__main__':

    # Create the attribute set by parsing the data-desc.txt file
    attributes = create_attribute_set();

    # Load the training examples
    dataset = load_training_examples()
    
    rootNode = ID3(dataset, attributes)
