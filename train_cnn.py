import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchmetrics
import os
import cnn

# Laver batches af dataset til træning og testing af cnn
def import_data(training_images, testing_images):
    training_dataloader = DataLoader(training_images, batch_size=1000)
    testing_dataloader = DataLoader(testing_images, batch_size=1000)

    return training_dataloader, testing_dataloader

# Instantierer et tomt cnn
#cnn = cnn.cnn()

# Opstiller et accuracy objekt til at måle hvor god modellen er (beregner andelen af korrekte fordsigelser ud af det totale antal data)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10) # num_classes er 10, da antallet af klasser(outputtet) i vores cnn er 10(cifrene 0-9)

# Bruger crossentropy som loss-funktion (beregner forskellen mellem modellens output-sandsynligheder og de faktiske tal)
loss_fn = nn.CrossEntropyLoss()

# Loop'er over 10 epoker, og udregner loss'et og accuracy for hvert.
def training_loop(training_dataloader, loss_fn, model):
    total_loss = 0
    accuracy.reset()
    size = len(training_dataloader)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, maximize=False) # Indstiller optimizeren, som opdaterer modellens parametre for at minimuere loss-funktionen

    for images,labels in training_dataloader:
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


def testing_loop(testing_dataloader, loss_fn, model):
    total_loss = 0
    accuracy.reset()
    size = len(testing_dataloader)
    with torch.no_grad():
        for images,labels in testing_dataloader:
            output = model(images)
            loss = loss_fn(output, labels)
            total_loss += loss
            accuracy.update(output,labels)
    avg_loss = total_loss / size
    print(f"Avg Testing Accuracy: {accuracy.compute() * 100 :.2f}%")
    print(f"Avg Testing Loss: {avg_loss}")


def train_test_and_save_model(model, epoches, filepath, training_images, testing_images):
    # Loader den angivne data og laver batches til træning og testing
    training_dataloader, testing_dataloader = import_data(training_images, testing_images)

    # Tester og træner over angivet epoker
    for i in range(epoches):
        print(f"Epoch: {i}")
        testing_loop(testing_dataloader, loss_fn, model)
        training_loop(training_dataloader, loss_fn, model)
        print("------------------")

    # Gemmer den trænede model i en fil og sletter den, hvis der allerede findes en fil gemt med den trænede model
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Den tidligere gemte cnn '{filepath}' er blevet slettet.")
    torch.save(model.state_dict(), filepath)
    print(f"Ny cnn gemt som '{filepath}'")