import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Baseret på #https://arxiv.org/pdf/1509.03456
# Preprocess Stack
# Brightness Equalization
        # Mangler formel til  ClipFactor
        # Bruger CLAHE : https://www.youtube.com/watch?v=tn2kmbUVK50  
        # https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
# Tjek Illumination
    # Hvis ensartet => Fortsæt
    # Hvis ikke => Adjust Illumination

# Grayscaling
# Unsharp Masking
# Ostu Binarization
# OCR

# 53.65% på DIDA dataset
def preprocess_stack(image):

    # Preprocess stack
    image = brightness_equalization(image)
    image = brightness_adjustment(image)
    image = grayscale(image)
    image = unsharp_mask(image)
    image = otsu_thresholding(image)

    # Inverterer farven
    image = 255 - image
    image = image.astype(np.uint8)
    return image

def seperate_digits(image):
    # ChatGPT
    # Find contours
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    digits = []
    bounding_boxes = get_bounding_boxes(contours)
    
    for box in bounding_boxes:
        padding = 10
        (x, y, w, h) = box
        digit = image[max(0, y - padding) : y+h+padding, max(0, x-padding) : x+w+padding]
        digits.append(digit)
    return digits, bounding_boxes

def get_bounding_boxes(contours):
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        # Filtrerer små boundingboxes fra.
        if 10 < w and 1 < h:
        # Gør det til en kvadrat for at beholde størrelsesforholdene ved resize.
            center_y = y + (h / 2)
            center_x = x + (w / 2)

            if h > w:
                w = h
            else:
                h = w 
    
            x = center_x - (w / 2)
            y = center_y - (h / 2)

            x = int(max(0, x))
            y = int(max(0, y))


            bounding_boxes.append((x, y, w, h))


    # Sortere således at listen går fra venstre mod højre.
    bounding_boxes.sort()
    return bounding_boxes


#Anvender CLAHE på value kanalen af billedet.
def brightness_equalization(image):
    # Formel til at bestemme clipLimit?
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))

    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # https://stackoverflow.com/questions/17185151/how-to-obtain-a-single-channel-value-image-from-hsv-image-in-opencv-2-1
    # Udfører CLAHE på value kanalen.
    h, s, v = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    v = clahe.apply(v)
    image[:, :, 2] = v
    image = cv.cvtColor(image, cv.COLOR_HSV2BGR)
    return image

def brightness_adjustment(image): # Brightness and contrast adjustment
    # Brug lidt mere tid her

    image_yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    y, _, _ = image_yuv[:, :, 0], image_yuv[:, :, 1], image_yuv[:, :, 2]

    max_brightness = y.max()
    target_brightness = max_brightness * 0.93
    avg_brightness = y.mean()

    alpha = 1.4  # Contrast gain
    beta = target_brightness - alpha * avg_brightness

    adjusted_y = np.clip(y * alpha + beta, 0, 255).astype(np.uint8)  # Adjust and clip
    image_yuv[:, :, 0] = adjusted_y
    
    image = cv.cvtColor(image_yuv, cv.COLOR_YUV2BGR)
    return image

def grayscale(image):
    # OpenCV BGR2GRAY anvender samme grayscalign teknik som i artiklen.
    # https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image

def unsharp_mask(image):
    # Juster til at passe med artikel
    # https://stackoverflow.com/questions/32454613/python-unsharp-mask
    gaussian_3 = cv.GaussianBlur(image, (0, 0), 2.0)
    unsharp_image = cv.addWeighted(image, 2.0, gaussian_3, -1.0, 0)
    return unsharp_image

def otsu_thresholding(image):
    #https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

    blur = cv.GaussianBlur(image,(5,5),0)
    _,image = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    return image
