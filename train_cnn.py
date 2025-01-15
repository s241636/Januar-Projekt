import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchmetrics
import os
import cnn

# Importere dataset til træning af et cnn
training_images = MNIST(root='data', transform=ToTensor(), train=True) # Datasættet bestående af billeder samt deres tilhørene tal
training_dataloader = DataLoader(training_images, batch_size=1000) # DataLoader pakker dataene i batches
testing_images = MNIST(root='data', transform=ToTensor(), train=False)
testing_dataloader = DataLoader(testing_images, batch_size=1000)

# Instantierer et tomt cnn
cnn = cnn.cnn()

# Opstiller et accuracy objekt til at måle hvor god modellen er (beregner andelen af korrekte fordsigelser ud af det totale antal data)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10) # num_classes er 10, da antallet af klasser(outputtet) i vores cnn er 10(cifrene 0-9)

# Bruger crossentropy som loss-funktion (beregner forskellen mellem modellens output-sandsynligheder og de faktiske tal)
# Indstiller optimizeren, som opdaterer modellens parametre for at minimuere loss-funktionen
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01, maximize=False)

# Loop'er over 10 epoker, og udregner loss'et og accuracy for hvert.
def training_loop(training_dataloader, optimizer, loss_fn):
    total_loss = 0
    accuracy.reset()
    size = len(training_dataloader)
    for images,labels in training_dataloader:
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


def testing_loop(testing_dataloader, loss_fn):
    total_loss = 0
    accuracy.reset()
    size = len(testing_dataloader)
    with torch.no_grad():
        for images,labels in testing_dataloader:
            output = cnn(images)
            loss = loss_fn(output, labels)
            total_loss += loss
            accuracy.update(output,labels)
    avg_loss = total_loss / size
    print(f"Avg Testing Accuracy: {accuracy.compute() * 100 :.2f}%")
    print(f"Avg Testing Loss: {avg_loss}")

for i in range(30):
    print(f"Epoch: {i}")
    testing_loop(testing_dataloader, loss_fn)
    training_loop(training_dataloader, optimizer, loss_fn)
    print("------------------")

# Gemmer den trænede model i en fil og sletter den, hvis der allerede findes en fil gemt med en trænede model
model_path = "trained_cnn.pth"

if os.path.exists(model_path):
    os.remove(model_path)
    print(f"Den tidligere gemte cnn '{model_path}' er blevet slettet.")
torch.save(cnn.state_dict(), model_path)
print(f"Ny cnn gemt som '{model_path}'")