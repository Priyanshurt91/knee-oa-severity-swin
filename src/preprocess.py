import cv2
import numpy as np

def preprocess_xray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    img = img / 255.0
    img = np.stack([img, img, img], axis=0)
    return img.astype(np.float32)
