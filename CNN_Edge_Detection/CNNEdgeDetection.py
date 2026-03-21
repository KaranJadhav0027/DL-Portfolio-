"""
===========================================================
Real-Time Image Classification using MobileNetV2 & OpenCV
===========================================================

Description:
------------
This script uses a pre-trained MobileNetV2 Convolutional 
Neural Network (CNN) from TensorFlow/Keras to perform 
real-time image classification from a webcam feed.

The script:
- Loads MobileNetV2 (cached using joblib for faster reuse).
- Captures video frames from the webcam.
- Preprocesses frames to match MobileNetV2 input size (224x224).
- Predicts the class of the object in real time.
- Displays the prediction label and confidence score on the frame.
- Exits when the user presses the 'q' key.

Dependencies:
-------------
- Python 3.x
- OpenCV (cv2)
- TensorFlow / Keras
- NumPy
- joblib

Usage:
------
Run the script:
    python realtime_classification.py

Press 'q' to quit the webcam window.
"""

import cv2
import numpy as np
import joblib
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)


###############################################################################################################
# Function name :- load_or_cache_model()
# Description :- Load MobileNetV2 model from cache if available, else create a new one and cache it
# Author :- Om Ravindra Wakhare
# Date :- 18/09/2025
###############################################################################################################
def load_or_cache_model(model_path="mobilenetv2_model.joblib"):
    """
    Load MobileNetV2 model from cache if available,
    else create a new one and cache it.
    """
    try:
        print("[INFO] Loading model from cache...")
        model = joblib.load(model_path)
    except Exception:
        print("[INFO] No cache found. Loading MobileNetV2 from Keras...")
        model = MobileNetV2(weights="imagenet")
        # Save model object for reuse
        joblib.dump(model, model_path)
        print(f"[INFO] Model cached at {model_path}")
    return model


###############################################################################################################
# Function name :- run_realtime_classification()
# Description :- Run real-time classification using webcam and MobileNetV2
# Author :- Om Ravindra Wakhare
# Date :- 18/09/2025
###############################################################################################################
def run_realtime_classification():
    """Run real-time classification using webcam and MobileNetV2."""
    # Load model (with caching)
    model = load_or_cache_model()

    # Initialize webcam capture
    cap = cv2.VideoCapture(0)

    print("[INFO] Starting webcam... Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture image.")
            break

        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to 224x224 (MobileNetV2 input size)
        img_resized = cv2.resize(img, (224, 224))

        # Expand dimensions to match input shape (1, 224, 224, 3)
        x = np.expand_dims(img_resized, axis=0).astype(np.float32)

        # Preprocess input for MobileNetV2
        x = preprocess_input(x)

        # Perform prediction
        preds = model.predict(x, verbose=0)

        # Decode prediction results (top-1)
        decoded = decode_predictions(preds, top=1)[0][0]
        label = f"{decoded[1]}: {decoded[2]*100:.1f}%"

        # Display prediction on the frame
        cv2.putText(frame, label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Show the output
        cv2.imshow("CNN Classification (MobileNetV2)", frame)

        # Break loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting...")
            break

    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_classification()
