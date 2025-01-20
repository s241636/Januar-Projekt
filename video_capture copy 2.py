# %%
import numpy as np
import torchvision
import torch.nn as nn
import torch
import cv2
import cnn 
import ImageProcessing as ip
import sympy

# %%
# Kalder funktionen der laver et cnn i filen cnn
cnn_math = cnn.math_cnn()
cnn_digits = cnn.cnn()

# Indlæser vægtene og bias fra forrigt trænede cnn
cnn_math.load_state_dict(torch.load("math_weights_10epochs.pth", weights_only=True))
cnn_math.eval()  # Sæt modellen i evalueringsmodus, således den ikke bruger teknikker, der kun anvendes under træning
cnn_digits.load_state_dict(torch.load("mnist_weights_10epochs.pth", weights_only=True))
cnn_digits.eval()  # Sæt modellen i evalueringsmodus, således den ikke bruger teknikker, der kun anvendes under træning
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
def calculate(pred):
    try:
        result = sympy.sympify(pred)
        return result 
    except sympy.SympifyError as e:
        return "Cannot be calculated"

def choose_model(digit_pred, math_pred):
    digit, math = digit_pred.argmax().item(), math_pred.argmax().item()
    digit_prop, math_prop = digit_pred[0][digit].item(), math_pred[0][math].item()

    digit_prop, math_prop = round(digit_prop, 4), round(math_prop, 4)
    
    if digit_prop > math_prop:
        return digit, digit_prop
    else:
        if math == 0:
            math = '+'
        if math == 1:
            math = '-'
        if math == 2:
            math = '*'
        return math, math_prop


# %%
while True:
    # Læser fra kameraet og viser dette i frame
    ret, frame = cam.read() 
    frame = cv2.flip(frame, 1) # Spejlvender kameraet

    # Tager udsnit af framen til boxen, markerer en box med tidligere definerede koordinater og viser i nyt vindue 
    cropped_frame = frame[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    cropped_frame = cv2.flip(cropped_frame, 1) # Spejlvender kameraet
    box_vid = cv2.resize(cropped_frame, box_size)
    box_vid = ip.preprocess_stack(box_vid)

    # Henter modellens genkendelse af tallet 
    image = ip.preprocess_stack(cropped_frame)
    digits, bounding_boxes = ip.seperate_digits(image)
    preds = []
    for d in digits:
        # Hvis bounding box'en faktisk har en størrelse gøres følgende
        if d.size:
            d = cv2.resize(d, (28,28))
            d = to_model_tensor(d)

            digit_pred, math_pred = sm(cnn_digits(d)), sm(cnn_math(d))

            # pred_digit = pred.argmax().item()
            # prop = pred[0][pred_digit].item()
            # prop = round(prop, 4)
            print(choose_model(digit_pred, math_pred))
            preds.append((choose_model(digit_pred, math_pred)))



    pred_digits = [pred[0] for pred in preds]
    # for idx, digit in enumerate(pred_digits):
    #     if digit == 10:
    #         pred_digits[idx] = '+'
    #     if digit == 11:
    #         pred_digits[idx] = '-'
    #     if digit == 12:
    #         pred_digits[idx] = '*'
    pred_string = ''.join(map(str, pred_digits))

    for bbox in bounding_boxes:
        (x, y, w, h) = bbox
        bbox_start = (x,y)
        bbox_end = (x + w, y + h)
        cv2.rectangle(box_vid, bbox_start, bbox_end, (255, 0, 0), 3)

    cv2.putText(frame, f"Prediction: {pred_string}:", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Calculation: {calculate(pred_string)}", (50,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.rectangle(frame, box[0], box[1], (0, 0, 0), 3)
    cv2.imshow('Debug', box_vid)
    cv2.imshow('Camera with Prediction', frame)

    # Lukker kamera-vinduet ved at trykke på 'esc'-knappen
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

