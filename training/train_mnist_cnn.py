# Importerer modules
import cnn
import NetUtils
import torch.nn as nn
import torch

# Loader test og træningsdata
training_loader = NetUtils.get_dataloader("MNIST", train=True, batch_size=10000)
testing_loader = NetUtils.get_dataloader("MNIST", train=False, batch_size=1000)

# Opretter det Neurale Netværk fra modellen
model_layers = cnn.mnist_only_cnn_v2()
net = NetUtils.NeuralNet(model_layers, nn.CrossEntropyLoss(), training_dataloader=training_loader, testing_dataloader=testing_loader)

# Træner og tester modellen, og gemmer den derefter.
epochs = 80
net.train_test_loop(epochs, print_info=True, save=True, savefolder="weights/mnistv2")
# filepath = f"mnistv2_weights_{epochs}epochs.pth"
# torch.save(net.model.state_dict(), filepath)

