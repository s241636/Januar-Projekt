# %%
import torch
import cnn
from torch.utils.data import DataLoader
import NetUtils

# %%
# Test accuracy on dataset
def test_accuracy_mnist(model, modelpath, dataset):
    model.load_state_dict(torch.load(modelpath)) # Load the trained model
    testing_dataloader = DataLoader(dataset, batch_size=1)
    
    correct_predictions = 0
    wrong_predictions = 0

    with torch.no_grad():
        for image, label in testing_dataloader:
            pred = model(image).argmax().item()

            if pred == label:
                correct_predictions += 1
            
            if pred != label:
                wrong_predictions += 1

    acc = (correct_predictions / (wrong_predictions + correct_predictions)) * 100
    return f"Accuracy: {acc:.2f}%"

# %%
# Test accuracy
model = cnn.cnn()
modelpath = "trained_cnn.pth"
dataset = NetUtils.get_dataset("MATH") # "MNIST", "DIDA" or "MATH"

test_accuracy_mnist(model, modelpath, dataset)


# %%
