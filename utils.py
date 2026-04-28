import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import streamlit as st

# Use Streamlit's cache to load the model just once.
# This prevents reloading the model every time the app re-runs.
@st.cache_resource
def load_detection_model():
    """Loads the pre-trained Keras model from the 'cnn_model.h5' file."""
    try:
        model = load_model('cnn_model.h5')
        model.summary()
        return model
    except Exception as e:
        st.error(f"Error: Could not load the model. Please ensure 'cnn_model.h5' is in the same directory. Details: {e}")
        return None

def detect_and_crop_face(img_cv):
    """
    Detects and crops a face from an image using MTCNN.
    Returns the cropped face (resized to 128x128) and a boolean indicating
    whether a face was successfully detected.
    """
    detector = MTCNN()
    results = detector.detect_faces(img_cv)
    
    if results:
        # Get the first and most confident face detection
        bounding_box = results[0]['box']
        x, y, width, height = bounding_box
        
        # Ensure the coordinates are within image boundaries
        x_end = x + width
        y_end = y + height
        face = img_cv[y:y_end, x:x_end]
        
        # Resize the face to the model's required input size
        face_resized = cv2.resize(face, (128, 128))
        return face_resized, True
    else:
        # If no face is detected, resize the entire image
        # The model will then try to classify based on the whole image.
        resized_img = cv2.resize(img_cv, (128, 128))
        return resized_img, False

def preprocess_for_prediction(face_img):
    """
    Preprocesses the face image for the CNN model's input.
    Converts to array, adds a batch dimension, and normalizes pixel values.
    """
    img_array = image.img_to_array(face_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to the range [0, 1]
    return img_array

def predict_real_or_fake(img):
    """
    Main function to analyze an image and predict if it's real or fake.
    
    Args:
        img (PIL.Image): The image to be analyzed.
        
    Returns:
        tuple: A tuple containing the prediction result ('Real' or 'Fake'),
               the confidence score (as a percentage), and a boolean
               indicating if a face was detected.
    """
    # Convert the PIL image to an OpenCV compatible format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Check if the image has three channels (RGB) before processing
    if len(img_cv.shape) < 3 or img_cv.shape[2] != 3:
        raise ValueError("Image must be a 3-channel (RGB) color image.")

    # Detect and crop the face from the image
    face, face_detected = detect_and_crop_face(img_cv)
    
    # Preprocess the cropped face for the model
    processed_image = preprocess_for_prediction(face)
    
    # Load the model and make the prediction
    model = load_detection_model()
    if model is None:
        raise RuntimeError("Model could not be loaded. Aborting prediction.")
        
    raw_prediction = model.predict(processed_image)
    
    # The model's output is a probability. Let's assume a higher value
    # indicates a higher probability of being a deepfake.
    probability_fake = float(raw_prediction[0][0])
    
    # Classify the image based on a 50% threshold
    if probability_fake >= 0.5:
        result = "Fake"
        # Confidence is the probability of the predicted class
        confidence = probability_fake * 100
    else:
        result = "Real"
        # Confidence is (1 - probability of the 'Fake' class)
        confidence = (1 - probability_fake) * 100
    
    return result, confidence, face_detected