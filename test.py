
# https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python
# Load datasets
from mnist import MNIST
mndata = MNIST('MNIST_ORG')
images, labels = mndata.load_testing()

# Construction of our neural network
    # 2 Hidden layers
    # 12 neuroner i hver
    # 10 output neuroner

# Input
# 10.000, 784 dimensionelle vektorer.
    # Inputlag => 784 neuroner

# 1. Hidden Layer
    # Input => 784 aktiveringsværdier
    # Output => 12 aktiveringsværdier

# 2. Hidden Layer
    # Input => 12 aktiveringsværdier
    # Output => 12 aktiveringsværdier

# Outputlag
    #Input => 12 aktiveringsværdier
    # Output => 10 neuroner(0-9), værdi mellem 0 og 1.

# https://pytorch.org/tutorials/beginner/basics/intro.html

import torch
import torch.nn as nn  # all neural network modules
import torch.optim as optim  # optimization algo
import torch.nn.functional as F  # Functions with no parameters -> activation functions
from torch.utils.data import DataLoader  # easier dataset management, helps create mini batches
import torchvision.datasets as datasets  # standard datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# Create our neural network by making a class that inherits from nn.Module, which is the base class for all neural networks in pytorch
# Then we specify this class to create a neural network that fits our problem
class NeuralNetwork(nn.Module):

    # Create the neural network in the constructor consisting of the input layer, the 2 hidden layers and the output layer
    def __init__(self, input = 28*28, layer_1 = 12, layer_2 = 12, output = 10):
        super().__init__()
        self.fc1 = nn.Linear(input, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.out = nn.Linear(layer_2, output)
    
    # Use activation function on all neurons in each layer
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.out(x))

        return x



# Draw an image
# Create a 2D array of grayscale values
data = np.array(images[1]).reshape(28, 28)  # 28x28 grid with values 0-255 representing a grey-scale

# Display the grid as an image
plt.imshow(data, cmap='grey', interpolation='nearest')
plt.axis('off')  # Turn off axis
plt.show()


# Create an instance of NeuralNetwork
neural_network = NeuralNetwork()

# Convert the images and labels to tensors
images = torch.Tensor(images)
print(neural_network(images[1]))
print(labels[1])

# Convert the tensors to a 10 dimensional vector (a list of 0's where the label's index is a 1)
converted_labels = []
for label in labels:
    vector = [0] * 10
    vector[label] = 1
    converted_labels.append(vector)
converted_labels = torch.Tensor(converted_labels)
print(converted_labels[1])

neural_network(images)
