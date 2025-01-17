# %%
import NetUtils
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchshow as ts

# %%
mnist = NetUtils.get_dataset("MNIST")
math = NetUtils.get_dataset("MATH")
samlet = ConcatDataset([mnist, math])
loader = DataLoader(samlet)
# %%
for idx, i in enumerate(loader):
    # if idx :
    #     break
    if i[1].item() not in [0,1,2,3,4,5,6,7,8,9]:
        ts.show(i[0])
        print(i[1].item())
        break

# %%
d = NetUtils.get_dataset("MATH")
loader = DataLoader(d, batch_size=1)
# %%
images, labels = NetUtils.load_data("MATH")
z = zip(images, labels)
z = list(z)
sorted_list = sorted(z, key=lambda x: x[1])


# %%
testset = []
for idx, i in enumerate(sorted_list):
    if idx % 18 == 0: 
        testset.append(i)

images = []
labels = []
for i in testset:
    images.append(i[0])
    labels.append(i[1])

images, labels = NetUtils.format_data_for_model(images, labels)
d = TensorDataset(images, labels)


# %%
d = NetUtils.get_dataset("MATH", train=False)
count_10 = 0
count_11 = 0
count_12 = 0
for i in d:
    if i[1].item() == 10:
        count_10 += 1
    if i[1].item() == 11:
        count_11 += 1
    if i[1].item() == 12:
        count_12 += 1

