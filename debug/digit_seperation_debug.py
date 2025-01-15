# %% Import
import torch
import torch.nn as nn
import torchvision
import torchshow as ts
import ImageProcessing as ip
import os
import cv2 as cv 
import mnist_cnn
import importlib
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as F


# %% Seperation Function
def seperate_digits(image):
    # ChatGPT
    # Find contours
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes
    bounding_boxes = []
    output_image = image.copy()

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        
        # Filtering small noise or very large boxes (optional)
        if 10 < w and 10 < h:
            bounding_boxes.append((x, y, w, h))
            cv.rectangle(output_image, (x, y), (x + w, y + h), (255, 255, 255), 2)  # Draw bounding box
            
    bounding_boxes.sort()
    digits = []
    for box in bounding_boxes:
        padding = 20
        (x, y, w, h) = box
        digit = image[y - padding : y+h+padding, x-padding:x+w+padding]


        digits.append(digit)
    return digits

# %% To tensor
def to_ts(image):
    image = torch.Tensor(image)
    image = image.float()/255
    return image

def resize(image):
    image = image.unsqueeze(0).unsqueeze(0)
    image = F.functional.resize((image), [28, 28], antialias=True)
    return image


# %%
net = mnist_cnn.cnn()
net.load_state_dict(torch.load('trained_cnn.pth', weights_only=True))

# %%
image = cv.imread('images_sequences/image1.png')
image = ip.preprocess_stack_v2(image)
plt.imshow(image, cmap="gray")

# %%
digits = ip.seperate_digits(image)
for d in digits:
    d = torch.Tensor(d)
    d = d.float()/255
    ts.show(d)


# %%
