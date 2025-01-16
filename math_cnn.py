# %%
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import cnn
import train_cnn
import os
from torchvision.io import read_image
import cv2
import matplotlib.pyplot as plt
import ImageProcessing as ip
import numpy as np
import torchmetrics

# %%
def load_math_symbols(symbol_folder):
    images = []
    labels = []
    image_paths = os.listdir(symbol_folder)
    for path in image_paths:
        path = f"{symbol_folder}/{path}"
        images.append(cv2.imread(path))
        labels.append(get_math_label(symbol_folder, path))
    images = [ip.preprocess_stack_v2(image) for image in images]
    images = np.expand_dims(images, 1)
    labels = np.array(labels)
    return images, labels


def get_math_label(symbol_folder, path):
    if path.startswith(f'{symbol_folder}/plus'):
        return 10
    if path.startswith(f'{symbol_folder}/minus'):
        return 11
    if path.startswith(f'{symbol_folder}/dot'):
        return 12
# %%
class MathDataset(Dataset):
    def __init__(self, images, labels, transform=ToTensor()):
        self.images, self.labels = images, labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

# %%
model = cnn.math_cnn()
epochs = 20
filepath = 'math_weights_20epochs.pth'
images, labels = load_math_symbols('data/symbols')
images = torch.Tensor(images)
labels = torch.Tensor(labels)
images = images.float()/255

training_images = images[0:3751]
testing_images = images[3751:]
training_labels = labels[0:3751]
testing_labels = labels[3751:]

training_labels = training_labels.type(torch.long)
testing_labels = testing_labels.type(torch.long)
# training_labels = training_labels.unsqueeze(0)
# testing_labels = testing_labels.unsqueeze(0)

training_dataset = MathDataset(training_images, training_labels)
training_loader = DataLoader(training_dataset)
testing_dataset = MathDataset(testing_images, testing_labels)
testing_loader = DataLoader(testing_dataset)

accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=13) # num_classes er 10, da antallet af klasser(outputtet) i vores cnn er 10(cifrene 0-9)
loss_fn = nn.CrossEntropyLoss()


# Training loop
total_loss = 0
accuracy.reset()
size = len(training_loader)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, maximize=False) # Indstiller optimizeren, som opdaterer modellens parametre for at minimuere loss-funktionen

for i in range(3):
    for images,labels in training_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        total_loss += loss
        loss.backward()
        optimizer.step()
        accuracy.update(output, labels)
    avg_loss = total_loss / size
    print(f"Avg Training Accuracy: {accuracy.compute() * 100:.2f}%")
    print(f"Avg Training Loss: {avg_loss}")
    # %%
