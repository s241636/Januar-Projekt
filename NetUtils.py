# %%
import os
import cv2
import ImageProcessing as ip
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torch
import numpy as np


DIDA_FOLDER = "data/DIDA"
MNIST_FOLDER = "data/MNIST"
MATH_FOLDER = "data/symbols"
SPLITKEY = 6
    # Splitkey definerer størrelsen af testsettet i forhold til data:
    #   Ved SPLITKEY = 6
    #   Testset = 1/6 af det originale dataset
    #   Træningsset = 5/6 af det originale dataset.

# %%
# Tager input 
def get_dataset(dataset, train=True, wholeset=False):
    if dataset != "MNIST":
        images, labels = load_data(dataset, train, wholeset)
        images, labels = format_data_for_model(images, labels)
        return TensorDataset(images, labels)
    elif dataset == "MNIST":
        return  MNIST(root='data', transform=ToTensor(), train=train)

def load_data(dataset, train=True, wholeset=False):
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
        return get_subset(images, labels, SPLITKEY, train=train)

def get_subset(images, labels, splitkey, train=True):
    testset = []
    trainingset = []

    data_list= list(zip(images, labels))
    
#   https://www.geeksforgeeks.org/python-program-to-sort-a-list-of-tuples-by-second-item/
    sorted_list = sorted(data_list, key=lambda x: x[1])

    for idx, i in enumerate(sorted_list):
        if idx % splitkey == 0: 
            testset.append(i)
        else:
            trainingset.append(i)

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
