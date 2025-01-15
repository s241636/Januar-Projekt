# %%
import numpy as np
import torchvision
import torch.nn as nn
import torch
import cv2
import cnn 
import ImageProcessing as ip

# %%
# Kalder funktionen der laver et cnn i filen cnn
cnn = cnn.cnn()

# Indlæser vægtene og bias fra forrigt trænede cnn
cnn.load_state_dict(torch.load("trained_cnn.pth", weights_only=True))
cnn.eval()  # Sæt modellen i evalueringsmodus, således den ikke bruger teknikker, der kun anvendes under træning
print("Model indlæst og klar til brug")
sm = nn.Softmax(dim=1)

# %%
# Laver kamera objekt
cam = cv2.VideoCapture(0)

# Henter default frame bredde og højde
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Laver box i midten af frame, som skal aflæse tallet
box_size = (200*2, 200) # Definerer tupel der aagiver størrelsen på boxen
box = [(int(frame_width // 2 - box_size[0] // 2), int(frame_height // 2 - box_size[1] // 2)),
       (int(frame_width // 2 + box_size[0] // 2), int(frame_height // 2 + box_size[1] // 2))] # Angiver boxens koordinater centreret i framen i en liste af tupels

# %%
def to_model_tensor(image):
    image = torch.Tensor(image)
    image = image.float()/255
    image = image.unsqueeze(0).unsqueeze(0)
    return image

# %%

while True:
    # Læser fra kameraet og viser dette i frame
    ret, frame = cam.read() 
    frame = cv2.flip(frame, 1) # Spejlvender kameraet

    # Tager udsnit af framen til boxen, markerer en box med tidligere definerede koordinater og viser i nyt vindue 
    cropped_frame = frame[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    cropped_frame = cv2.flip(cropped_frame, 1) # Spejlvender kameraet
    box_vid = cv2.resize(cropped_frame, box_size)
    box_vid = ip.preprocess_stack_v2(box_vid)

    # Henter modellens genkendelse af tallet 
    image = ip.preprocess_stack_v2(cropped_frame)
    digits, bounding_boxes = ip.seperate_digits(image)
    preds = []
    for d in digits:
        # Hvis bounding box'en faktisk har en størrelse gøres følgende
        if d.size:
            d = cv2.resize(d, (28,28))
            d = to_model_tensor(d)
            pred = sm(cnn(d))
            pred_digit = pred.argmax().item()
            prop = pred[0][pred_digit].item()
            prop = round(prop, 4)
            preds.append((pred_digit, prop))



    pred_digits = [pred[0] for pred in preds]
    pred_string = ''.join(map(str, pred_digits))

    for bbox in bounding_boxes:
        (x, y, w, h) = bbox
        bbox_start = (x,y)
        bbox_end = (x + w, y + h)
        cv2.rectangle(box_vid, bbox_start, bbox_end, (255, 0, 0), 3)

    cv2.putText(frame, f"Prediction: {pred_string}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.rectangle(frame, box[0], box[1], (0, 0, 0), 3)
    cv2.imshow('Cropped frame', box_vid)
    cv2.imshow('Camera with Prediction', frame)

    # Lukker kamera-vinduet ved at trykke på 'esc'-knappen
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()