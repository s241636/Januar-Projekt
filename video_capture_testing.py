# %%
import torch.nn as nn
import torch
import cv2
import cnn 
import ImageProcessing as ip
import sympy
import pandas as pd
import os


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

frame_count = 0
count_limit = 10
results = []
sample_data = []
label = ""
space_bar_pressed = False

def save_results(results, label):
    right_answers = 0

    for result in results:
        if result.strip() == label.strip():
            right_answers += 1
    
    accuracy = right_answers / len(results) * 100

    return f"{accuracy} %"

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
            pred, prop = choose_model(digit_pred, math_pred)
            if prop > 0.75:
                preds.append((choose_model(digit_pred, math_pred)))

    pred_digits = [pred[0] for pred in preds]
    pred_string = ''.join(map(str, pred_digits))

    for bbox in bounding_boxes:
        (x, y, w, h) = bbox
        bbox_start = (x,y)
        bbox_end = (x + w, y + h)
        cv2.rectangle(box_vid, bbox_start, bbox_end, (255, 0, 0), 3)


    cv2.putText(frame, f"Prediction: {pred_string} = {calculate(pred_string)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Label: {label}", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    if space_bar_pressed:
        cv2.putText(frame, f"SAVED RESULTS", (50,350), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.rectangle(frame, box[0], box[1], (0, 0, 0), 3)
    cv2.imshow('Debug', box_vid)
    cv2.imshow('Camera with Prediction', frame)

    key = cv2.waitKey(1)

    # Gemmer resultater fra 10 frames ved at trykke på 'space'-knappen
    if key == 32 and frame_count == 0:
        frame_count = count_limit 
        results = []
        space_bar_pressed = True
 
    if frame_count > 0:
        results.append(pred_string) 
        frame_count -= 1 

        if frame_count == 0:
            accuracy = save_results(results, label)
            sample_data.append([accuracy, label, results])
            space_bar_pressed = False
            print("Accuracy:", accuracy, "\Label:", label, "\nResults:",results, )
            print("-------------------------------------------------------------------------------------------------\n")
    
     
    
    # Lukker kamera-vinduet ved at trykke på 'esc'-knappen
    if key == 27: 
        break
    # Sletter tegn i label ved at trykke på 'back-space'-knappen
    elif key == 127:
        if len(label) > 0:
            label = label[:-1]
    # Tilføjer tegn til label
    elif 33 <= key <= 126:  
        label += chr(key)


# Save sample_data in excel file
file_path = 'Experiment_data.xlsx'
sheet_name = 'test_3'

df = pd.DataFrame(sample_data, columns=['Accuracy', "Label", "Sample-data"])

if not os.path.exists(file_path):
    df.to_excel(file_path, sheet_name=sheet_name, index=False)
    print(f"File {file_path} created with sheet '{sheet_name}'!")
else:
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Sheet '{sheet_name}' added to the existing file {file_path}!")

cv2.destroyAllWindows()


