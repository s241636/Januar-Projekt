# %%
# Importerer modules
import cnn
import NetUtils
import torch.nn as nn
import torch
import os
import cv2
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import time


# %%
# Loader test og træningsdata
# training_loader = NetUtils.get_dataloader("MATH", train=True, batch_size=3750)
# testing_loader = NetUtils.get_dataloader("MATH", train=False, batch_size=750)
PLUS_PATH = "data/symbols_extended/plus"
MINUS_PATH = "data/symbols_extended/minus"

plus_paths = os.listdir(PLUS_PATH)
if ".DS_Store" in plus_paths:
    plus_paths.remove(".DS_Store")
minus_paths = os.listdir(MINUS_PATH)
if ".DS_Store" in minus_paths:
    minus_paths.remove(".DS_Store")

images = []
labels = []

for path in plus_paths:
    images.append(cv2.imread(f"{PLUS_PATH}/{path}"))
    labels.append(0)
for path in minus_paths:
    images.append(cv2.imread(f"{MINUS_PATH}/{path}"))
    labels.append(1)

# %%
trainingset, testset = NetUtils.get_subset(images, labels, 7)
training_images, training_labels = zip(*trainingset)
test_images, test_labels = zip(*testset)

training_images, training_labels = NetUtils.format_data_for_model(training_images, training_labels)
test_images, test_labels = NetUtils.format_data_for_model(test_images, test_labels)


training_dataset = TensorDataset(training_images, training_labels)
training_dataset = TensorDataset(training_images, training_labels)
test_dataset = TensorDataset(test_images, test_labels)  

training_loader = DataLoader(training_dataset, batch_size=1000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=500, shuffle=True)


# %%
# Opretter det Neurale Netværk fra modellen
model_layers = cnn.math_only_cnn()
net = NetUtils.NeuralNet(model_layers, nn.CrossEntropyLoss(), training_dataloader=training_loader, testing_dataloader=test_loader)

# Træner og tester modellen, og gemmer den derefter.
epochs = 10
net.train_test_loop(epochs, print_info=True)
filepath = f"math_weights_{epochs}epochs.pth"
torch.save(net.model.state_dict(), filepath)
