import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return img[y:y+h, x:x+w]
    else:
        return None

def extract_features(face_img):
    # Resize the image to a standard size
    face_img = cv2.resize(face_img, (100, 100))
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve contrast
    equalized = cv2.equalizeHist(gray)
    
    # Flatten the image into a 1D array
    features = equalized.flatten()
    
    return features