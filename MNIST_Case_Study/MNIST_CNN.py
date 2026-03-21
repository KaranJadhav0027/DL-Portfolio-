"""
================================================================================
MNIST Handwritten Digit Recognition using Convolutional Neural Network (CNN)
================================================================================

Author  : Swamit Amit Jadhav
Purpose : This script implements a complete CNN pipeline for MNIST handwritten 
          digit recognition including data loading, model training, evaluation,
          and prediction with model persistence for future use.

Workflow:
---------
1. Load and preprocess MNIST dataset (60,000 training + 10,000 test images)
2. Build a custom CNN architecture optimized for 28x28 grayscale images
3. Train the model with data augmentation and early stopping
4. Evaluate model performance on test set
5. Save trained model and training history
6. Provide prediction interface for new images
7. Visualize training progress and sample predictions

Requirements:
-------------
- Python 3.x
- TensorFlow / Keras (`pip install tensorflow`)
- NumPy (`pip install numpy`)
- Matplotlib (`pip install matplotlib`)
- Joblib (`pip install joblib`)

Usage:
------
Run the script directly:
    python MNIST_CNN.py

The script will automatically:
- Download MNIST dataset if not available
- Train the CNN model
- Save the trained model
- Display training progress and results

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


###############################################################################################################
# Function name :- load_and_preprocess_data()
# Description :- Load MNIST dataset and preprocess it for CNN training
# Author :- Swamit Amit Jadhav
# Date :- 18/09/2025
###############################################################################################################
def load_and_preprocess_data():
    """Load and preprocess MNIST dataset for CNN training."""
    print("[INFO] Loading MNIST dataset...")
    
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print(f"[INFO] Training set: {x_train.shape[0]} samples")
    print(f"[INFO] Test set: {x_test.shape[0]} samples")
    print(f"[INFO] Image shape: {x_train.shape[1:]} (28x28 grayscale)")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN input (samples, height, width, channels)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to categorical (one-hot encoding)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print(f"[INFO] Preprocessed training data shape: {x_train.shape}")
    print(f"[INFO] Preprocessed test data shape: {x_test.shape}")
    print(f"[INFO] Training labels shape: {y_train.shape}")
    print(f"[INFO] Test labels shape: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)


###############################################################################################################
# Function name :- build_cnn_model()
# Description :- Build a custom CNN architecture optimized for MNIST digit recognition
# Author :- Swamit Amit Jadhav
# Date :- 18/09/2025
###############################################################################################################
def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """Build a custom CNN model for MNIST digit recognition."""
    print("[INFO] Building CNN model architecture...")
    
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"[INFO] Model built successfully!")
    print(f"[INFO] Total parameters: {model.count_params():,}")
    
    return model


###############################################################################################################
# Function name :- create_data_augmentation()
# Description :- Create data augmentation pipeline for improved model generalization
# Author :- Swamit Amit Jadhav
# Date :- 18/09/2025
###############################################################################################################
def create_data_augmentation():
    """Create data augmentation pipeline for training."""
    print("[INFO] Creating data augmentation pipeline...")
    
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ])
    
    return data_augmentation


###############################################################################################################
# Function name :- train_model()
# Description :- Train the CNN model with callbacks and data augmentation
# Author :- Swamit Amit Jadhav
# Date :- 18/09/2025
###############################################################################################################
def train_model(model, x_train, y_train, x_val, y_val, epochs=50, batch_size=128):
    """Train the CNN model with callbacks and monitoring."""
    print(f"[INFO] Starting model training for {epochs} epochs...")
    
    # Create data augmentation
    data_augmentation = create_data_augmentation()
    
    # Create callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'mnist_best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        data_augmentation(x_train),
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("[INFO] Training completed!")
    return history


###############################################################################################################
# Function name :- evaluate_model()
# Description :- Evaluate model performance on test set with detailed metrics
# Author :- Swamit Amit Jadhav
# Date :- 18/09/2025
###############################################################################################################
def evaluate_model(model, x_test, y_test):
    """Evaluate model performance on test set."""
    print("[INFO] Evaluating model on test set...")
    
    # Get predictions
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"[INFO] Test Accuracy: {test_accuracy:.4f}")
    print(f"[INFO] Test Loss: {test_loss:.4f}")
    
    # Classification report
    print("\n[INFO] Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes))
    
    return test_accuracy, test_loss, y_pred_classes, y_true_classes


###############################################################################################################
# Function name :- plot_training_history()
# Description :- Plot training history including accuracy and loss curves
# Author :- Swamit Amit Jadhav
# Date :- 18/09/2025
###############################################################################################################
def plot_training_history(history):
    """Plot training history for accuracy and loss."""
    print("[INFO] Plotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


###############################################################################################################
# Function name :- plot_confusion_matrix()
# Description :- Plot confusion matrix for model evaluation
# Author :- Swamit Amit Jadhav
# Date :- 18/09/2025
###############################################################################################################
def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix for model evaluation."""
    print("[INFO] Plotting confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix - MNIST Digit Recognition')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


###############################################################################################################
# Function name :- visualize_predictions()
# Description :- Visualize sample predictions with true and predicted labels
# Author :- Swamit Amit Jadhav
# Date :- 18/09/2025
###############################################################################################################
def visualize_predictions(x_test, y_true, y_pred, num_samples=16):
    """Visualize sample predictions."""
    print(f"[INFO] Visualizing {num_samples} sample predictions...")
    
    # Select random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        axes[i].imshow(x_test[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'True: {y_true[idx]}, Pred: {y_pred[idx]}', 
                         color='green' if y_true[idx] == y_pred[idx] else 'red')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


###############################################################################################################
# Function name :- save_model_and_history()
# Description :- Save trained model and training history for future use
# Author :- Swamit Amit Jadhav
# Date :- 18/09/2025
###############################################################################################################
def save_model_and_history(model, history, test_accuracy):
    """Save model and training history."""
    print("[INFO] Saving model and training history...")
    
    # Save model
    model.save('mnist_cnn_model.h5')
    print("[INFO] Model saved as 'mnist_cnn_model.h5'")
    
    # Save training history
    history_data = {
        'history': history.history,
        'test_accuracy': test_accuracy,
        'timestamp': datetime.now().isoformat()
    }
    joblib.dump(history_data, 'training_history.joblib')
    print("[INFO] Training history saved as 'training_history.joblib'")


###############################################################################################################
# Function name :- load_trained_model()
# Description :- Load pre-trained model for predictions
# Author :- Swamit Amit Jadhav
# Date :- 18/09/2025
###############################################################################################################
def load_trained_model(model_path='mnist_cnn_model.h5'):
    """Load pre-trained model for predictions."""
    if os.path.exists(model_path):
        print(f"[INFO] Loading pre-trained model from {model_path}")
        model = keras.models.load_model(model_path)
        return model
    else:
        print(f"[ERROR] Model file {model_path} not found!")
        return None


###############################################################################################################
# Function name :- predict_digit()
# Description :- Predict digit from a single image
# Author :- Swamit Amit Jadhav
# Date :- 18/09/2025
###############################################################################################################
def predict_digit(model, image):
    """Predict digit from a single image."""
    if len(image.shape) == 2:
        image = image.reshape(1, 28, 28, 1)
    elif len(image.shape) == 3 and image.shape[-1] != 1:
        image = image.reshape(1, 28, 28, 1)
    
    # Normalize image
    image = image.astype('float32') / 255.0
    
    # Make prediction
    prediction = model.predict(image, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return predicted_digit, confidence


###############################################################################################################
# Function name :- main()
# Description :- Main function to orchestrate the complete MNIST training pipeline
# Author :- Swamit Amit Jadhav
# Date :- 18/09/2025
###############################################################################################################
def main():
    """Main function to run the complete MNIST training pipeline."""
    print("=" * 80)
    print("MNIST Handwritten Digit Recognition using CNN")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Split training data into train and validation
    val_split = 0.2
    val_size = int(len(x_train) * val_split)
    
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:]
    y_train = y_train[val_size:]
    
    print(f"[INFO] Training samples: {len(x_train)}")
    print(f"[INFO] Validation samples: {len(x_val)}")
    print(f"[INFO] Test samples: {len(x_test)}")
    
    # Build model
    model = build_cnn_model()
    
    # Display model architecture
    print("\n[INFO] Model Architecture:")
    model.summary()
    
    # Train model
    history = train_model(model, x_train, y_train, x_val, y_val, epochs=50)
    
    # Evaluate model
    test_accuracy, test_loss, y_pred, y_true = evaluate_model(model, x_test, y_test)
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred)
    visualize_predictions(x_test, y_true, y_pred)
    
    # Save model and history
    save_model_and_history(model, history, test_accuracy)
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
