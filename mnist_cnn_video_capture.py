import numpy as np
import torchvision
import torch.nn as nn
import torch
import cv2
import mnist_cnn 

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
box_size = (200, 200) # Definerer tupel der aangiver størrelsen på boxen
box = [(int(frame_width // 2 - box_size[0] // 2), int(frame_height // 2 - box_size[1] // 2)),
       (int(frame_width // 2 + box_size[0] // 2), int(frame_height // 2 + box_size[1] // 2))] # Angiver boxens koordinater centreret i framen i en liste af tupels


# Funktion der forbereder et billede til billedgenkendelse
def prep_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255
    image = 1.0 - image # Inverterer image (sort til hvis og omvendt)
    
    image = torch.tensor(image, dtype=torch.float32) # Konverterer til PyTorch tensor
    image = image.unsqueeze(0).unsqueeze(0) # Tilføjer dimensioner for batch_size og kanaler (reshaping til (1, 1, 28, 28))
    return image


while True:
    # Læser fra kameraet og viser dette i frame
    ret, frame = cam.read() 
    frame = cv2.flip(frame, 1) # Spejlvender kameraet

    # Tager udsnit af framen til boxen, markerer en box med tidligere definerede koordinater og viser i nyt vindue 
    cropped_frame = frame[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    cropped_frame = cv2.flip(cropped_frame, 1) # Spejlvender kameraet
    box_vid = cv2.resize(cropped_frame, (200,200))
    cv2.imshow('Cropped frame', box_vid)

    # Henter modellens genkendelse af tallet 
    image = cv2.resize(cropped_frame, (28,28)) # Resizer kvadratet fra framen til 28x28, således den passer til modellens input
    image = prep_image(image)
    # Finder modellens bud på talgenkendelsen
    pred = cnn(image)
    num_pred = pred.argmax().item()
    # Beregner hvor sikker modellen er på talgenkendelsen
    sm = nn.Softmax(dim=1)
    prop = sm(pred)
    prop = prop[0, num_pred]
    print(f"Number prediction: {num_pred} \nProbability: {prop}\n")

    # Laver label der viser modellens tal-genkendelse
    cv2.putText(frame, f"Prediction: ", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Probability: ", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

    if prop > 0.75:
        cv2.putText(frame, f"Prediction: {num_pred}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Probability: {prop}", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.rectangle(frame, box[0], box[1], (0, 0, 0), 3)
    cv2.imshow('Camera with Prediction', frame)

    # Lukker kamera-vinduet ved at trykke på 'esc'-knappen
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

