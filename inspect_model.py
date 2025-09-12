import tensorflow as tf
import os

# Define the path to your .h5 file
model_path = 'cnn_model.h5'

# Check if the file exists before trying to load it
if not os.path.exists(model_path):
    print(f"Error: The model file '{model_path}' was not found.")
else:
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        print("--- Model loaded successfully! ---")
        print("\n--- Model Summary ---")
        model.summary()

    except Exception as e:
        print(f"An error occurred while loading the model: {e}")