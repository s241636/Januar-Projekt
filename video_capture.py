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
mnist_net = cnn.mnist_only_cnn_v2()
math_net = cnn.math_only_cnn()
classifier_net = cnn.classifier_cnn()

# Indlæser vægtene og bias fra forrigt trænede cnn
mnist_net.load_state_dict(torch.load("weights/mnistv2/ep63loss0.05538863688707352", weights_only=True))
math_net.load_state_dict(torch.load("weights/math_weights_10epochs.pth", weights_only=True))
classifier_net.load_state_dict(torch.load("weights/classifier_weights_20epochs.pth", weights_only=True))


mnist_net.eval()  # Sæt modellen i evalueringsmodus, således den ikke bruger teknikker, der kun anvendes under træning
math_net.eval()  # Sæt modellen i evalueringsmodus, således den ikke bruger teknikker, der kun anvendes under træning
classifier_net.eval()

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


# %%
while True:
    # Læser fra kameraet og viser dette i frame
    ret, frame = cam.read() 
    frame = cv2.flip(frame, 1) # Spejlvender kameraet

    # Tager udsnit af framen til boxen, markerer en box med tidligere definerede koordinater og viser i nyt vindue 
    cropped_frame = frame[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    cropped_frame = cv2.flip(cropped_frame, 1) # Spejlvender kameraet
    box_vid = ip.preprocess_stack(cropped_frame)

    # Henter modellens genkendelse af tallet 
    image = ip.preprocess_stack(cropped_frame)
    digits, bounding_boxes = ip.seperate_digits(image)
    preds = []
    display_images = []

    for idx, d in enumerate(digits):
        # Hvis bounding box'en faktisk har en størrelse gøres følgende
        if d.size:
            d = cv2.resize(d, (28,28))
            display_images.append(cv2.resize(d, (112,112)))
            d = to_model_tensor(d)
                
            classifier_pred = sm(classifier_net(d))
            classifier_pred_class = classifier_pred.argmax().item()
            classifier_prob = classifier_pred[0][classifier_pred_class].item()

            if classifier_prob < 0.8:    
                mnist_pred = sm(mnist_net(d))
                mnist_pred_digit = mnist_pred.argmax().item()
                mnist_prop = mnist_pred[0][mnist_pred_digit].item()

                math_pred = sm(math_net(d))
                math_pred_digit = math_pred.argmax().item()                
                math_prop = math_pred[0][math_pred_digit].item()    
            
                if math_pred_digit == 0:
                    math_pred_digit = '+'
                elif math_pred_digit == 1:
                    math_pred_digit = '-'
                elif math_pred_digit == 2:
                    math_pred_digit = '*'


                # Edge case hvor et 4-tal observeres som et plus af matematik modellen.
                # Har observeret at matematik modellen er oversikker på at 4-taller er +'er, men MNIST modellen laver ikke samme fejl.
                # Derfor kan dette korrigeres ved at sænke sikkerheden på matematik modellen hvis der er tvivl om det er et 4-tal eller +.
                # Hvis det er et plus er der nemlig sjældent tvivl fra hverken classifieren eller MNIST modellen, og tvivlen vil typisk
                # Indikerer at det faktisk er et 4-tal.
                if mnist_pred_digit == 4 and math_pred_digit == '+':
                    math_prop -= 0.05


                if math_prop > mnist_prop:
                    preds.append((math_pred_digit, math_prop))
                else:
                    preds.append((mnist_pred_digit, mnist_prop))




            # Hvis objektet er et tal
            elif classifier_pred_class == 0:
                mnist_pred = sm(mnist_net(d))
                mnist_pred_digit = mnist_pred.argmax().item()
                mnist_prop = mnist_pred[0][mnist_pred_digit].item()
                preds.append((mnist_pred_digit, mnist_prop))

            # Hvis objektet er et matematisk symbol
            elif classifier_pred_class == 1:
                math_pred = sm(math_net(d))
                math_pred_digit = math_pred.argmax().item()                
                math_prop = math_pred[0][math_pred_digit].item()    
                
                if math_pred_digit == 0:
                    math_pred_digit = '+'
                elif math_pred_digit == 1:
                    math_pred_digit = '-'
                elif math_pred_digit == 2:
                    math_pred_digit = '*'

                preds.append((math_pred_digit, math_prop))
    
    if len(display_images):
        display_images = np.concatenate(display_images, axis=1)
        cv2.imshow('Model input', display_images)


    pred_digits = [pred[0] for pred in preds]
    pred_string = ''.join(map(str, pred_digits))

    for bbox in bounding_boxes:
        (x, y, w, h) = bbox
        bbox_start = (x,y)
        bbox_end = (x + w, y + h)
        cv2.rectangle(box_vid, bbox_start, bbox_end, (255, 0, 0), 3)

    cv2.putText(frame, f"Prediction: {pred_string}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Calculation: {calculate(pred_string)}", (50,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(box_vid, f"Digit count: {len(digits)}", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, box[0], box[1], (0, 0, 0), 3)
    cv2.imshow('Debug', box_vid)
    cv2.imshow('Camera with Prediction', frame)

    # Lukker kamera-vinduet ved at trykke på 'esc'-knappen
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

