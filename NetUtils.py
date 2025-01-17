# %%
import os
import cv2
import ImageProcessing as ip
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
import torch


DIDA_FOLDER = "data/DIDA"
MNIST_FOLDER = "data/MNIST"
MATH_FOLDER = "data/symbols"

# Fra en mappe, load billeddata
    # Skal kunne skelne mellem forskellige dataset, og hvordan labels skal udledes.

# %%
# Takes a folder of images, returns list of images (0-255) and of labels.
def get_dataset(dataset):
    if dataset != "MNIST":
        images, labels = load_data(dataset)
        images, labels = format_data_for_model(images, labels)
        return TensorDataset(images, labels)
    elif dataset == "MNIST":
        return  MNIST(root='data', transform=ToTensor(), train=True)


def load_data(dataset):
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
    
    return images, labels
    
def format_data_for_model(images, labels):
    images = [ip.preprocess_stack(image) for image in images]
    for idx, image in enumerate(images):
        if image.shape != (28, 28):
            images[idx] = cv2.resize(image, (28,28))   

    images, labels = listdata_to_tensor(images, labels)
    return images, labels

def listdata_to_tensor(images, labels):
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