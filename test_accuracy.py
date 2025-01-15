# %%
import torch
import torch.nn as nn
import torchvision
import torchshow as ts
import ImageProcessing as ip
import os
import cv2 as cv
import mnist_cnn
import importlib
import torchvision.transforms as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


# %%
def load_dida_images_with_labels(dida_folder):
    image_paths = os.listdir(dida_folder)
    if ".DS_Store" in image_paths:
        image_paths.remove(".DS_Store")
    labels = []
    images = []
    for path in image_paths:
        label = path[0]
        labels.append(int(label))
        image = cv.imread(f"DIDA/{path}")
        images.append(image)
    return images, labels

def to_model_tensor(image):
    image = torch.Tensor(image)
    image = image.float()/255
    image = image.unsqueeze(0).unsqueeze(0)
    return image

# %%
# Test model on DIDA Dataset
net = mnist_cnn.cnn()
net.load_state_dict(torch.load('trained_cnn.pth', weights_only=True))
import mnist_cnn
importlib.reload(mnist_cnn) 

images, labels = load_dida_images_with_labels("DIDA")
image_count = len(images)
test_set = zip(images,labels)
correct_predictions = 0
for idx, (image, label) in enumerate(test_set):
    image = ip.preprocess_stack_v2(image)
    image = to_model_tensor(image)
    image = F.functional.resize(image, [28, 28], antialias=True)
    pred = net(image).argmax().item()
    if pred == label:
        correct_predictions += 1
acc = (correct_predictions / image_count) * 100
print(f"DIDA: V1 Accuracy: {acc:.2f}%")


# %%
# Test model on MNIST Dataset without dropout
net = mnist_cnn.cnn()
net.load_state_dict(torch.load('trained_cnn(no dropout).pth', weights_only=True))
import mnist_cnn
importlib.reload(mnist_cnn) 
testing_images = MNIST(root='data', transform=ToTensor(), train=False)
testing_dataloader = DataLoader(testing_images, batch_size=1)
correct_predictions = 0
for idx, (image, label) in enumerate(testing_dataloader):
    label = label.item()
    pred = net(image).argmax().item()
    if pred == label:
        correct_predictions += 1
acc = (correct_predictions / image_count) * 100
print(f"MNIST: V1 Accuracy: {acc:.2f}%")

# %%
# Test model on MNIST Dataset with dropout
net = mnist_cnn.cnn_dropout()
net.load_state_dict(torch.load('trained_cnn(with dropout).pth', weights_only=True))
import mnist_cnn
importlib.reload(mnist_cnn) 
testing_images = MNIST(root='data', transform=ToTensor(), train=False)
testing_dataloader = DataLoader(testing_images, batch_size=1)
correct_predictions = 0
for idx, (image, label) in enumerate(testing_dataloader):
    label = label.item()
    pred = net(image).argmax().item()
    if pred == label:
        correct_predictions += 1
acc = (correct_predictions / image_count) * 100
print(f"MNIST: V1 Accuracy: {acc:.2f}%")

# %%
# Test for confusion matrix on MNIST Dataset
net = mnist_cnn.cnn()
net.load_state_dict(torch.load('trained_cnn(no dropout).pth', weights_only=True))
import mnist_cnn
importlib.reload(mnist_cnn) 
testing_images = MNIST(root='data', transform=ToTensor(), train=False)
testing_dataloader = DataLoader(testing_images, batch_size=1)
correct_predictions = 0

for i in range(10):
    correct_predictions = 0
    wrong_predictions = 0

    for idx, (image, label) in enumerate(testing_dataloader):
        label = label.item()
        pred = net(image).argmax().item()

        if pred == label and label == i:
            correct_predictions += 1

        if pred != label and label == i:
            wrong_predictions += 1
        
    acc = (correct_predictions / (correct_predictions + wrong_predictions)) * 100
    print(f"{i}: {acc:.2f}%")





# %%
