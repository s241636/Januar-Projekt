import mnist_cnn 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import numpy as np

# Kalder funktionen der laver et cnn i filen mnist_cnn
cnn = mnist_cnn.cnn()

# Indlæser vægtene og bias fra forrigt trænede cnn
cnn.load_state_dict(torch.load("trained_cnn.pth", weights_only=True))
cnn.eval()  # Sæt modellen i evalueringsmodus, således den ikke bruger teknikker, der kun anvendes under træning
print("Model indlæst og klar til brug")

# Laver kamera objekt
cam = cv2.VideoCapture(0)

# Henter default frame bredde og højde
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Laver box i midten af frame, som skal aflæse tallet
box_size = (60, 60) # Definerer tupel der aangiver størrelsen på boxen
box = [(int(frame_width // 2 - box_size[0] // 2), int(frame_height // 2 - box_size[1] // 2)),
       (int(frame_width // 2 + box_size[0] // 2), int(frame_height // 2 + box_size[1] // 2))] # Angiver boxens koordinater centreret i framen i en liste af tupels

while True:
    # Læser fra kameriet og viser dette i frame
    ret, frame = cam.read() 
    frame = cv2.flip(frame, 1) # Spejlvender kameraet
    cv2.imshow('Camera', frame)

    # Tager udsnit af framen til boxen, markerer en box med tidligere definerede koordinater og viser i nyt vindue 
    cropped_frame =  frame[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    box_vid = cv2.resize(cropped_frame, (200,200))
    cv2.imshow('Cropped frame', box_vid)

    # Henter modellens genkendelse af tallet 
    image = cv2.resize(cropped_frame, (28,28))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255
    image = torch.tensor(image, dtype=torch.float32) # Image til PyTorch tensor
    image = image.unsqueeze(0).unsqueeze(0) # Tilføjer dimensionen for batch_size og kanaler (reshaping til (1, 1, 28, 28))
    pred = cnn(image)
    print(pred.argmax().item())

    # Laver label der viser modellens tal-genkendelse
    cv2.putText(frame, f"Prediction: {pred.argmax().item()}", (frame_width // 4, frame_height // 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Lukker kamera-vinduet ved at trykke på 'esc'-knappen
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()




# # Afprøver modellen på et givent index af billederne.
# img_idx = 10
# pred = net(training_images[img_idx][0].unsqueeze(0)) 
# print("Model output:")
# print(pred)
# sm = nn.Softmax(dim=1)
# print("Efter softmax:")
# print(sm(pred))
# print(f"Modul bud: {pred.argmax()}")
# print()
# ts.show(training_images[img_idx][0])