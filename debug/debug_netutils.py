# %%
import NetUtils
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchshow as ts
import random

# %%
math_data = NetUtils.get_dataset("MATH", train=True)

# %%
d = list(math_data)
# %%
img = random.choice(d)[0]
ts.show(img)
# %%
