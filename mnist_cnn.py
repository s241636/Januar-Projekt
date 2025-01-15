import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchmetrics
import os
import mnist_cnn


# Importere dataset til træning af et cnn
training_images = MNIST(root='data', transform=ToTensor(), train=True) # Datasættet bestående af billeder samt deres tilhørene tal
training_dataloader = DataLoader(training_images, batch_size=1000) # DataLoader pakker dataene i batches
testing_images = MNIST(root='data', transform=ToTensor(), train=False)
testing_dataloader = DataLoader(testing_images, batch_size=1000)

# Instantierer et tomt cnn
cnn = mnist_cnn.cnn()

# Opstiller et accuracy objekt til at måle hvor god modellen er (beregner andelen af korrekte fordsigelser ud af det totale antal data)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10) # num_classes er 10, da antallet af klasser(outputtet) i vores cnn er 10(cifrene 0-9)

# Bruger crossentropy som loss-funktion (beregner forskellen mellem modellens output-sandsynligheder og de faktiske tal)
# Indstiller optimizeren, som opdaterer modellens parametre for at minimuere loss-funktionen
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01, maximize=False)

        # 6. lag: Fully Connected Layer
            # Tager 1D vektoren fra 5. lag, som er i 250 dimensioner, og producerer 10 output der hver repræsenterer et tal fra 0-9
        nn.Linear(250,10), # input er nu 5 x 5 x 10
    )
    return cnn


