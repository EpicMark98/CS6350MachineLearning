import numpy as np
import random

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


# Class representing a single neuron
class Neuron:
    def __init__(self, prevLayerWidth, randomInit=False, sigmoid=True):
        self.output = 0
        self.gradient = 0
        if prevLayerWidth != None:
            self.weightGrads = np.zeros(prevLayerWidth)
        else:
            self.weightGrads = None
        self.useSigmoid = sigmoid

        if prevLayerWidth != None:
            # Initialize weights
            self.weights = np.zeros(prevLayerWidth)
            if randomInit:
                for i in range(prevLayerWidth):
                    self.weights[i] = random.gauss()

    # Calculates new output using the weights and the activation on the prevLayer outputs
    def updateOutput(self, inputVals):
        result = np.dot(inputVals, self.weights)

        # Handle the different activations
        if self.useSigmoid:
            temp = 1 + np.exp(-1 * result)
            self.output = 1 / temp
        else:
            self.output = result

    # Calculates neuron weight gradients
    def calcWeightGrads():



# Class representing a layer
class Layer:
    def __init__(self, size, prevLayer, sigmoid=True):
        self.width = size
        self.prevLayer = prevLayer
        self.useSigmoid = sigmoid
        self.neurons = [None] * self.width
        for i in range(self.width):
            if prevLayer != None:
                self.neurons[i] = Neuron(prevLayer.width, False, self.useSigmoid)
            else:
                self.neurons[i] = Neuron(None, False, self.useSigmoid)

        # Hardcode the constant feature 1 to be the first neuron
        self.neurons[0].output = 1

    # Puts an example as the values of the layer
    def setOutput(self, ex):
        if self.prevLayer != None:
            print("Attempted to set output on a non-input layer!")
            return
        for i in range(self.width-1):
            self.neurons[i+1].output = ex[i]   # Do not change first neuron output

    # Returns all layer outputs as a vector
    def getOutputs(self):
        result = np.zeros(self.width)
        for i in range(self.width):
            result[i] = self.neurons[i].output
        return result

    # Updates all neuron outputs
    def updateOutputs(self):
        inputVals = self.prevLayer.getOutputs()
        if self.width == 1:
            self.neurons[0].updateOutput(inputVals)
        for i in range(self.width-1):
            self.neurons[i+1].updateOutput(inputVals)

    # Calculate all gradients
    def calcNewGradients(self, prevLayer, nextLayer):
        # Calculate neuron gradients
        for i in range(1, self.width):
            grad = 0
            for j in range(nextLayer.width):
                currNeuron = nextLayer.neurons[j]
                grad += currNeuron.gradient * currNeuron.output * (1-currNeuron.output) * currNeuron.weights[i]
            self.neurons[i].gradient = grad

        # Calculate weight gradients
        for i in range(1, self.width):
            self.neurons[i].calcWeightGrads()


# Class representing a network
class Network:
    def __init__(self, numHidden, width, inputSize):
        self.numHidden = numHidden
        self.width = width
        self.layers = [None] * (numHidden + 2)

        # Initialize layers
        self.layers[0] = Layer(inputSize, None, False)
        for i in range(numHidden):
            self.layers[i+1] = Layer(width, self.layers[i])
        self.layers[-1] = Layer(1, self.layers[-2], False)

    # Given an example (without the label), performs the forward pass
    def forwardPass(self, example):
        self.layers[0].setOutput(example)
        for i in range(0, self.numHidden):
            self.layers[i+1].updateOutputs()
        self.layers[-1].updateOutputs()
        return self.layers[-1].neurons[0].output

    # Update gradient values using backpropagation
    def backpropagation(self, label):
        # Update last layer gradient
        self.layers[-1].neurons[0].gradient = self.layers[-1].neurons[0].output - label

        # Update hidden layer gradients
        for i in range(0, self.numHidden):
            self.layers[self.numHidden-i].calcNewGradients(self.layers[(self.numHidden-i)-1], self.layers[(self.numHidden-i)+1])

# Wrapper for main
def main():
    # Load examples
    #trainingData = load_examples()
    #testData = load_examples("test.csv")

    # Instantiate the network
    network = Network(2, 3, 3)

    # Hardcode the weights for the test example
    network.layers[1].neurons[1].weights = np.array([-1, -2, -3])
    network.layers[1].neurons[2].weights = np.array([1, 2, 3])
    network.layers[2].neurons[1].weights = np.array([-1, -2, -3])
    network.layers[2].neurons[2].weights = np.array([1, 2, 3])
    network.layers[3].neurons[0].weights = np.array([-1, 2, -1.5])

    testEx = np.array([1, 1, 1, 1])

    result = network.forwardPass(testEx[:-1])

    print(result)

if __name__ == '__main__':
        main()