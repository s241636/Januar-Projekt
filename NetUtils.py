# %%
import os
import cv2
import ImageProcessing as ip
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torchmetrics
import torch
import numpy as np
import random
import torchshow as ts
import torch.nn as nn
import cnn

DIDA_FOLDER = "data/DIDA"
MNIST_FOLDER = "data/MNIST"
MATH_FOLDER = "data/symbols"
SPLITKEY = 6
NUM_CLASSES = 3

    # Splitkey definerer størrelsen af testsættet i forhold til data:
    #   Ved SPLITKEY = 6
    #   Testset = 1/6 af det originale datasæt
    #   Træningssæt = 5/6 af det originale datasæt.

# %%
class NeuralNet():
    def __init__(self, model, loss_fn, training_dataloader, testing_dataloader):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, maximize=False)
        self.training_dataloader = training_dataloader
        self.testing_dataloader = testing_dataloader
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.softmax_fn = nn.Softmax()

    def training_loop(self, print_info=False):
            self.accuracy.reset()
            total_loss = 0
            size = len(self.training_dataloader)
            for images,labels in self.training_dataloader:
                self.optimizer.zero_grad()
                # output = self.softmax_fn(self.model(images))
                output = self.model(images)
                loss = self.loss_fn(output, labels)
                # print(output)
                # print(labels)
                # print(output.shape)
                # print(labels.shape)
                # self.accuracy.update(output, labels)
                total_loss += loss
                loss.backward()
                self.optimizer.step()
            if print_info:
                print("Training")
                print(f"Total loss: {total_loss}")
                print(f"Avg loss: {total_loss / size}")
                # print(f"Avg Training Accuracy: {self.accuracy.compute() * 100:.2f}%")

                # print(f"Output: \n")
                # print(f"{output}")
                print()

    
    def testing_loop(self, print_info=False):
            self.accuracy.reset()
            total_loss = 0
            size = len(self.testing_dataloader)
            with torch.no_grad():
                for images,labels in self.testing_dataloader:
                    # output = self.softmax_fn(self.model(images))
                    output = self.model(images)
                    # self.accuracy.update(output, labels)
                    loss = self.loss_fn(output, labels)
                    total_loss += loss
            if print_info:
                print(f"Testing")
                print(f"Total Loss: {total_loss}")
                print(f"Avg Loss: {total_loss / size}")
                # print(f"Avg Accuracy: {self.accuracy.compute() * 100:.2f}%")

                # print(f"Output: \n")
                # print(f"{output}")
                print("--------------")

    def train_test_loop(self, epochs, print_info=False):
        for epoch in range(epochs):
            if print_info:
                print(f"Epoch: {epoch}")
            self.training_loop(print_info=print_info)
            self.testing_loop(print_info=print_info)



# %%
def get_dataloader(dataset, batch_size=1, train=True, wholeset=False):
    return DataLoader(get_dataset(dataset, train, wholeset), batch_size=batch_size)

# Tager input 
def get_dataset(dataset, train=True, wholeset=False):
    if dataset in ["DIDA", "MATH"]:
        images, labels = load_data(dataset, train, wholeset)
        images, labels = format_data_for_model(images, labels)
        return TensorDataset(images, labels)

    elif dataset == "MNIST" and not wholeset:
        return  MNIST(root='data', transform=ToTensor(), train=train)
    elif dataset == "MNIST" and wholeset:
        return ConcatDataset([MNIST(root='data', 
        transform=ToTensor(), train=False), MNIST(root='data', transform=ToTensor(), train=True)])

    elif dataset == "MNIST_MATH":
        return ConcatDataset([get_dataset("MNIST", train=train), get_dataset("MATH", train=train)])

def load_data(dataset, train=True, wholeset=False):
    foldername = ''
    if dataset == "MATH":
        label_fn = get_math_label
        foldername = MATH_FOLDER
    elif dataset == "DIDA":
        label_fn = get_dida_label
        foldername = DIDA_FOLDER
    
    filenames = os.listdir(foldername)
    if ".DS_Store" in filenames:
        filenames.remove(".DS_Store")
    
    images = []
    labels = []

    for file in filenames:
        file_path = f"{foldername}/{file}"
        images.append(cv2.imread(file_path)) 
        labels.append(label_fn(file))

    # Hvis dataset over alt data ønskes returneres det.
    if wholeset:
        return images, labels
    else:
        images, labels = get_subset(images, labels, SPLITKEY, train=train)
        return images, labels

def get_subset(images, labels, splitkey, train=True):
    testset = []
    trainingset = []

    data_list= list(zip(images, labels))

    # https://www.geeksforgeeks.org/python-program-to-sort-a-list-of-tuples-by-second-item/
    sorted_list = sorted(data_list, key=lambda x: x[1])

    for idx, i in enumerate(sorted_list):
        if idx % splitkey == 0: 
            testset.append(i)
        else:
            trainingset.append(i)

    # https://www.geeksforgeeks.org/python-unzip-a-list-of-tuples/
    if train:
        return zip(*trainingset)

    if not train:
        return zip(*testset)
    

def format_data_for_model(images, labels):
    images = [ip.preprocess_stack(image) for image in images]
    for idx, image in enumerate(images):
        if image.shape != (28, 28):
            images[idx] = cv2.resize(image, (28,28))   

    images, labels = listdata_to_tensor(images, labels)
    return images, labels

def listdata_to_tensor(images, labels):
    labels = np.array(labels)
    images = np.array(images)
    labels = torch.tensor(labels)
    images = torch.tensor(images)
    images = images.float()/255
    images = images.unsqueeze(1)

    return images, labels

def get_math_label(filename):
    if filename.startswith(f'plus'):
        return 10
    elif filename.startswith(f'minus'):
        return 11
    elif filename.startswith(f'dot'):
        return 12
    return -1

def get_dida_label(filename):
    return int(filename[0])

