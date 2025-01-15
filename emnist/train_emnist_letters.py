import torch
import torch.nn as nn
from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchshow as ts
import torchmetrics
import matplotlib.pyplot as plt
import os
import emnist_letters_cnn

# Importere dataset, kun træning indtil videre.
training_images_letters = EMNIST( root='data' , split = 'letters', transform = ToTensor(), download = True, train = True, target_transform=lambda y: y - 1)
training_dataloader_letters = DataLoader(training_images_letters, batch_size=1000, shuffle= True)

testing_images_letters = EMNIST( root='data' , split = 'letters', transform = ToTensor(), download = True, train = False, target_transform=lambda y: y - 1)
testing_dataloader_letters = DataLoader(testing_images_letters, batch_size=1000)

cnn = emnist_letters_cnn.cnn_letters()

# Opstiller et accuracy objekt til at måle hvor god modellen er.
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=26) #num_classes er 10, da antallet af klasser(outputtet) i vores cnn er 10(cifrene 0-9)

# Bruger crossentropy til at udregne losset, og indstiller optimizeren.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01, maximize=False)

# Loop'er over 10 epoker, og udregner loss'et og accuracy for hvert.
def training_loop(training_dataloader_letters, optimizer, loss_fn):
    total_loss = 0
    accuracy.reset()
    size = len(training_dataloader_letters)
    for images,labels in training_dataloader_letters:
        optimizer.zero_grad()
        output = cnn(images)
        loss = loss_fn(output, labels)
        total_loss += loss
        loss.backward()
        optimizer.step()
        accuracy.update(output, labels)
    avg_loss = total_loss / size
    print(f"Avg Training Accuracy: {accuracy.compute() * 100:.2f}%")
    print(f"Avg Training Loss: {avg_loss}")

def testing_loop(testing_dataloader_letters, loss_fn):
    total_loss = 0
    accuracy.reset()
    size = len(testing_dataloader_letters)
    with torch.no_grad():
        for images,labels in testing_dataloader_letters:
            output = cnn(images)
            loss = loss_fn(output, labels)
            total_loss += loss
            accuracy.update(output,labels)
    avg_loss = total_loss / size
    print(f"Avg Testing Accuracy: {accuracy.compute() * 100 :.2f}%")
    print(f"Avg Testing Loss: {avg_loss}")

for i in range(30):
    print(f"Epoch: {i}")
    testing_loop(testing_dataloader_letters, loss_fn)
    training_loop(training_dataloader_letters, optimizer, loss_fn)
    print("------------------")

# Gemmer den trænede model i en fil og sletter den, hvis der allerede findes en fil gemt med en trænede model
model_path = "trained_cnn_letters.pth"

if os.path.exists(model_path):
    os.remove(model_path)
    print(f"Den tidligere gemte cnn '{model_path}' er blevet slettet.")
torch.save(cnn.state_dict(), model_path)
print(f"Ny cnn gemt som '{model_path}'")