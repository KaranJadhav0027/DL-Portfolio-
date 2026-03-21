
import cv2
import numpy as np
import joblib
import os
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

def load_or_save_model(model_path="mobilenetv2_model.pkl"):
    """Load model from disk if available, else create and save it."""
    if os.path.exists(model_path):
        print("Loading model from disk...")
        model = joblib.load(model_path)
    else:
        print("Downloading MobileNetV2 model (ImageNet weights)...")
        model = MobileNetV2(weights="imagenet")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    return model


def MarvellousImageClassifier():
    # 1) Load model (from disk if available)
    model = load_or_save_model()

    # 2) Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit the webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # 3) Preprocess for MobileNetV2: BGR -> RGB, resize to 224x224, scale
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (224, 224))
        x = np.expand_dims(img_resized, axis=0).astype(np.float32)
        x = preprocess_input(x)

        # 4) Predict
        preds = model.predict(x, verbose=0)
        decoded = decode_predictions(preds, top=1)[0][0]  # (class_id, class_name, score)
        label = f"{decoded[1]}: {decoded[2]*100:.1f}%"

        # 5) Overlay prediction on the frame
        cv2.putText(frame, label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Show output
        cv2.imshow("Real-time CNN Classification (MobileNetV2)", frame)

        # 6) Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7) Cleanup
    cap.release()
    cv2.destroyAllWindows()

def main():
    MarvellousImageClassifier()


if __name__ == "__main__":
    main()
