"""Image recogion demo."""
# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import ImageProcessing as ip
import mnist_cnn
# %%
# Set up neural network
model = mnist_cnn.cnn()
model.load_state_dict(torch.load('trained_cnn.pth', weights_only=True))


# %%
# Set up figure
fig = plt.figure(1)
ax = fig.gca()
vid = plt.imshow(np.ones((224, 224, 3)))
lbl = plt.text(0, 0, "Loading...", size=25, va='top')
lbl.set_bbox({'facecolor': 'white', 'alpha': 0.5, 'edgecolor': 'none'})
ax.set_axis_off()
plt.ion()
plt.show()

# %%
# Set up video capture
cap = cv2.VideoCapture(0)
fig.canvas.mpl_connect('close_event', lambda evt: cap.release())


def update(i):
    """Capture image and update figure with prediction."""
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        raise Exception("Unable to get image from webcam")
    min_dim = np.min(frame.shape[0:2])
    frame = cv2.resize(frame[0:min_dim, 0:min_dim, :], (224, 224))

    
    predicted_digits = []
    


    digits = ip.preprocess_stack_v2(frame)
    label = ''
    if len(digits):
        for digit in digits:
            predicted_digits.append(model(digit).argmax())
    print(predicted_digits)

    # Display image and label in figure
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    vid.set_data(frame)
    lbl.set_text(label)

    return vid, lbl


# Run animation loop
ani = FuncAnimation(fig, update, blit=True, interval=100)

# Keep running until user quits
input('Press any key to quit.')
# %%
