import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import cnn

# download training data from the FashionMNISTdataset.
training_data = datasets.MNIST(
    train=True,
    transform=ToTensor(),
    download=True,
    root="data"
)

# download test data from the FashionMNIST dataset.
test_data = datasets.MNIST(
    train=False,
    transform=ToTensor(),
    download=True,
    root="data"
)


batch_size = 64

# create data loaders
training_loader = DataLoader(training_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=64)

for X, y in test_loader:
  print(f"Shape of X [N C H W]: {X.shape}")
  print(f"Shape of y: {y.shape} {y.dtype}")
  break


# get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Load model and move to device
model = cnn.cnn_v2()
model.load_state_dict(torch.load('trained_cnn_v2.pth', map_location=device, weights_only=True))
model = model.to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print(f"size: {size}")
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device, dtype=torch.float32)  # Ensure input is float32
        y = y.to(device)  # Move labels to the correct device

        # compute predicted y by passing X to the model
        prediction = model(X)

        # compute the loss
        loss = loss_fn(prediction, y)

      #  apply zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  

        # print training progress
        if batch % 100 == 0:
            loss_value = loss.item()  
            current = batch * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            prediction = model(X)
            test_loss += loss_fn(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epoch = 30
for t in range(epoch):
  print(f"Epoch {t+1}\n-------------------------------")
  train(training_loader, model, loss_fn, optimizer)
  test(test_loader, model, loss_fn)
print("Done!")
