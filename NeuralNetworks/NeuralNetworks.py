import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.init as init
import pandas as pd


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
                    self.weights[i] = random.gauss(0, 1)

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
    def calcWeightGrads(self, prevLayer):
        for i in range(len(self.weightGrads)):
            if self.useSigmoid:
                self.weightGrads[i] = prevLayer.neurons[i].output * self.output * (1-self.output) * self.gradient
            else:
                self.weightGrads[i] = prevLayer.neurons[i].output * self.gradient

    # Updates weights based on gradients and learning rate
    def step(self, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= self.weightGrads[i] * lr


# Class representing a layer
class Layer:
    def __init__(self, size, prevLayer, sigmoid, randInit):
        self.width = size
        self.prevLayer = prevLayer
        self.useSigmoid = sigmoid
        self.neurons = [None] * self.width
        for i in range(self.width):
            if prevLayer != None:
                self.neurons[i] = Neuron(prevLayer.width, randInit, self.useSigmoid)
            else:
                self.neurons[i] = Neuron(None, randInit, self.useSigmoid)

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
                if nextLayer.width > 1 and j == 0:
                    continue
                currNeuron = nextLayer.neurons[j]
                if nextLayer.useSigmoid:
                    grad += currNeuron.gradient * currNeuron.output * (1-currNeuron.output) * currNeuron.weights[i]
                else:
                    grad += currNeuron.gradient * currNeuron.weights[i]
            self.neurons[i].gradient = grad

        # Calculate weight gradients
        for i in range(1, self.width):
            self.neurons[i].calcWeightGrads(prevLayer)

    # Updates weights based on gradients
    def step(self, lr):
        for i in range(1, len(self.neurons)):
            self.neurons[i].step(lr)

        # Handle last layer
        if self.width == 1:
            self.neurons[0].step(lr)


# Class representing a network
class Network:
    def __init__(self, numHidden, width, inputSize, randInit):
        self.numHidden = numHidden
        self.width = width
        self.layers = [None] * (numHidden + 2)

        # Initialize layers
        self.layers[0] = Layer(inputSize, None, False, randInit)
        for i in range(numHidden):
            self.layers[i+1] = Layer(width, self.layers[i], True, randInit)
        self.layers[-1] = Layer(1, self.layers[-2], False, randInit)

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
        # Calculate weight gradients
        for i in range(0, self.layers[-1].width):
            self.layers[-1].neurons[i].calcWeightGrads(self.layers[-2])

        # Update hidden layer gradients
        for i in range(0, self.numHidden):
            self.layers[self.numHidden-i].calcNewGradients(self.layers[(self.numHidden-i)-1], self.layers[(self.numHidden-i)+1])

    # Update the weights based on the gradients and learning rate
    def step(self, lr):
        for i in range(1, len(self.layers)):
            self.layers[i].step(lr)

    # Calculates prediction error given data with labels
    def calcError(self, data):
        numWrong = 0
        for ex in data:
            x_i = ex[:-1]
            y_i = ex[-1]

            pred = self.forwardPass(x_i)
            if (pred >= 0 and y_i == -1):
                numWrong += 1
            elif (pred < 0 and y_i == 1):
                numWrong += 1

        return numWrong / len(data)

# Wrapper for main
def main():
    width = int(input("Enter a hidden layer width: "))
    initial = input("Choose the weight initialization method (\"zeros\" or \"random\"): ")

    # Load examples
    trainingData = load_examples()
    testData = load_examples("test.csv")

    # Instantiate the network
    network = Network(2, width, len(trainingData[0])-1, initial=="random")

    # Hyperparameters
    epochs = 40
    gamma0 = 0.05
    a = 1

    # Learn
    if initial == "random":
        print("Learning with network of width " + str(width) + " and random weight initialization")
    else:
        print("Learning with network of width " + str(width) + " and zero weight initialization")
    for e in range(epochs):
        print("Epoch " + str(e+1) + "/" + str(epochs))
        # Update the learning rate
        lr = gamma0 / (1 + (gamma0 / a) * e)

        np.random.shuffle(trainingData)
        for ex in trainingData:
            x_i = ex[:-1]
            y_i = ex[-1]

            network.forwardPass(x_i)
            network.backpropagation(y_i)
            network.step(lr)

    print("Training Error: " + str(network.calcError(trainingData)))
    print("Test Error: " + str(network.calcError(testData)))

# Dataset for PyTorch to use
class NNDataset(Dataset):
    def __init__(
        self,
        csv_file="train.csv"
    ):
        # read csv file
        data_csv = pd.read_csv(csv_file)
        self.X = torch.tensor(data_csv.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(data_csv.iloc[:, -1].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_loop(dataloader, model, loss_fn, optimizer):
    """
    trains your model for an epoch
    returns an array of loss values over the training epoch
    """
    # set model to train mode
    model.train()
    losses = list()

    # the model can run on either CPU or CUDA (also supports AMD's ROCm)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # move images to GPU if needed
        X, y = X.to(device), y.to(device)

        # zero gradients from previous step
        optimizer.zero_grad()

        # compute prediction and loss
        pred = model(X) # Remember forward()? This calls that.
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()

    return losses


def test_loop(dataloader, model, loss_fn):
    """
    tests your model on the test set
    returns average MSE
    """
    # the model can run on either CPU or CUDA (also supports AMD's ROCm)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # set model to eval mode
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            # move images to GPU if needed
            X, y = X.to(device), y.to(device)

            # compute prediction and loss
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    return test_loss

class RELU_NN(nn.Module):
    def __init__(self, num_features, width, depth):
        super(RELU_NN, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(num_features, width))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

        # Apply he initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        x = self.network(x)
        return x

class TANH_NN(nn.Module):
    def __init__(self, num_features, width, depth):
        super(TANH_NN, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(num_features, width))
        layers.append(nn.Tanh())

        # Hidden layers
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

        # Apply Xavier initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        x = self.network(x)
        return x

def pytorch_nn():
    # Get user parameters
    activation = input("Please enter the activation function to use ('t' for tanh or 'r' for relu): ")
    width = int(input("Enter the width: "))
    depth = int(input("Enter the depth: "))

    # Load the data
    train_dataset = NNDataset("train.csv")
    test_dataset = NNDataset("test.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define model
    model = None
    if activation == 'r':
        print("Using ReLU with width " + str(width) + " and depth " + str(depth))
        model = RELU_NN(4, width, depth)
    else:
        print("Using tanh with width " + str(width) + " and depth " + str(depth))
        model = TANH_NN(4, width, depth)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train the network
    epochs = 40
    for e in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss_val = loss(outputs, y_batch)
            loss_val.backward()
            optimizer.step()
            total_loss += loss_val.item()
        print(f"Epoch [{e+1}/{epochs}], Loss: {total_loss / len(train_dataloader):.4f}")

    # Test
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for X_batch, y_batch in test_dataloader:
            outputs = model(X_batch).squeeze()
            loss_val = loss(outputs, y_batch)
            total_loss += loss_val.item()
        print(f"Test Loss: {total_loss / len(test_dataloader):.4f}")
    

if __name__ == '__main__':
    print("NEURAL NETWORKS")
    print("Please choose which part you want to run.")
    print("1 - Manual implementation")
    print("2 - PyTorch (bonus question)")
    print("3 - Quit")
    choice = input("Enter your choice: ")

    if choice == '1':
        main()
    elif choice == '2':
        pytorch_nn()