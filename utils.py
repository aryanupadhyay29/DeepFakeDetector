import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import streamlit as st

@st.cache_resource
def load_detection_model():
    
    return load_model('cnn_model.h5')

# Function to detect and crop the face using MTCNN
def detect_and_crop_face(img):
    detector = MTCNN()
    results = detector.detect_faces(img)
    if results:
        bounding_box = results[0]['box']
        x, y, width, height = bounding_box
        face = img[y:y+height, x:x+width]
        face = cv2.resize(face, (128, 128))  # Resize to the target size
        return face, True
    else:
        # If no face is detected, return the original resized image
        return cv2.resize(img, (128, 128)), False

# Function to preprocess the cropped face
def preprocess_face(face):
    img_array = image.img_to_array(face)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  
    return img_array

# Function to predict if an image is real or fake
def predict_real_or_fake(img):
    
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    
    face, face_detected = detect_and_crop_face(img_cv)
    processed_image = preprocess_face(face)
    
    # Load model and predict
    model = load_detection_model()
    prediction = model.predict(processed_image)
    
   
    result = 'Real' if prediction[0][0] < 0.5 else 'Fake'
    confidence = float(abs(0.5 - prediction[0][0]) * 2 * 100) 
    
    return result, confidence, face_detected
