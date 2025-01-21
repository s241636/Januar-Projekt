# Importerer modules
import cnn
import NetUtils
import torch.nn as nn
import torch


# Loader test og træningsdata
training_loader = NetUtils.get_dataloader("MATH", train=True, batch_size=1000)
testing_loader = NetUtils.get_dataloader("MATH", train=False, batch_size=300)

# Opretter det Neurale Netværk fra modellen
model_layers = cnn.math_cnn()
net = NetUtils.NeuralNet(model_layers, nn.CrossEntropyLoss(), training_dataloader=training_loader, testing_dataloader=testing_loader)

# Træner og tester modellen, og gemmer den derefter.
epochs = 8
net.train_test_loop(epochs, print_info=True)
filepath = f"math_weights_{epochs}epochs.pth"
torch.save(net.model.state_dict(), filepath)

