# Importerer modules
import cnn
import NetUtils
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

# Loader test og træningsdata
mnist_data = NetUtils.get_dataset("MNIST", train=False)
math_data = NetUtils.get_dataset("MATH", train=True)

datset = ConcatDataset([mnist_data, math_data])
training_loader = DataLoader(dataset=datset, batch_size=1000)

# training_loader = NetUtils.get_dataloader("MNIST", train=True, batch_size=10000)
# testing_loader = NetUtils.get_dataloader("MNIST", train=False, batch_size=1000)

# Opretter det Neurale Netværk fra modellen
model_layers = cnn.mnist_math_cnn()
net = NetUtils.NeuralNet(model_layers, nn.CrossEntropyLoss(), training_dataloader=training_loader)

# Træner og tester modellen, og gemmer den derefter.
epochs = 30
for i in range(epochs):
    net.training_loop(print_info=True)

filepath = f"mnist_math_weights_{epochs}epochs.pth"
torch.save(net.model.state_dict(), filepath)

