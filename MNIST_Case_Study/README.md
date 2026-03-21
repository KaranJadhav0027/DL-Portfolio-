# MNIST Handwritten Digit Recognition using Convolutional Neural Network (CNN)

## Overview
This project implements a comprehensive deep learning solution for MNIST handwritten digit recognition using a custom Convolutional Neural Network (CNN) architecture. The application provides end-to-end functionality including data preprocessing, model training, evaluation, visualization, and prediction capabilities with professional-grade features.

## Features
- **Complete CNN Pipeline**: End-to-end implementation from data loading to prediction
- **Custom Architecture**: Optimized CNN architecture for 28x28 grayscale images
- **Data Augmentation**: Advanced data augmentation for improved generalization
- **Model Persistence**: Save and load trained models for future use
- **Comprehensive Evaluation**: Detailed metrics, confusion matrix, and visualizations
- **Training Monitoring**: Callbacks for early stopping, model checkpointing, and learning rate reduction
- **Professional Visualization**: Training curves, sample predictions, and confusion matrix plots

## Technical Specifications
- **Dataset**: MNIST (60,000 training + 10,000 test images)
- **Input Size**: 28x28 grayscale images
- **Classes**: 10 digits (0-9)
- **Framework**: TensorFlow/Keras
- **Architecture**: Custom CNN with Batch Normalization and Dropout
- **Optimization**: Adam optimizer with learning rate scheduling

## Prerequisites
- Python 3.7+
- 8GB+ RAM recommended
- GPU support optional but recommended for faster training
- Internet connection for MNIST dataset download

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SwamitJadhav/Deep_Learning_Models.git
cd Dl_casestudies/MNIST\ Handwritten\ Digit\ Recognition/
```

### 2. Install Dependencies
```bash
pip install -r ../../requirements.txt
```

### 3. Verify Installation
```bash
python MNIST_CNN.py
```

## Usage

### Basic Usage
```bash
python MNIST_CNN.py
```

### Expected Output
The script will automatically:
1. Download MNIST dataset (if not available)
2. Preprocess and split the data
3. Build and train the CNN model
4. Evaluate performance on test set
5. Generate visualizations and save results
6. Save the trained model for future use

### Training Process
- **Epochs**: 50 (with early stopping)
- **Batch Size**: 128
- **Validation Split**: 20%
- **Data Augmentation**: Rotation, zoom, and translation
- **Callbacks**: Early stopping, model checkpointing, learning rate reduction

## Architecture

### CNN Model Architecture
```
Input (28, 28, 1)
├── Conv2D(32) + BatchNorm + ReLU
├── Conv2D(32) + BatchNorm + ReLU
├── MaxPooling2D(2,2) + Dropout(0.25)
├── Conv2D(64) + BatchNorm + ReLU
├── Conv2D(64) + BatchNorm + ReLU
├── MaxPooling2D(2,2) + Dropout(0.25)
├── Conv2D(128) + BatchNorm + ReLU + Dropout(0.25)
├── Flatten
├── Dense(512) + BatchNorm + Dropout(0.5)
├── Dense(256) + Dropout(0.5)
└── Dense(10) + Softmax
```

### Core Components
1. **Data Loading**: `load_and_preprocess_data()`
   - MNIST dataset loading and preprocessing
   - Normalization and reshaping for CNN input
   - One-hot encoding of labels

2. **Model Building**: `build_cnn_model()`
   - Custom CNN architecture design
   - Batch normalization and dropout layers
   - Model compilation with Adam optimizer

3. **Data Augmentation**: `create_data_augmentation()`
   - Random rotation, zoom, and translation
   - Improved model generalization

4. **Training**: `train_model()`
   - Model training with callbacks
   - Early stopping and model checkpointing
   - Learning rate reduction on plateau

5. **Evaluation**: `evaluate_model()`
   - Comprehensive performance metrics
   - Classification report and confusion matrix
   - Test accuracy and loss calculation

6. **Visualization**: Multiple plotting functions
   - Training history curves
   - Confusion matrix heatmap
   - Sample prediction visualization

## Performance Metrics
- **Expected Accuracy**: 99%+ on test set
- **Training Time**: ~30-60 minutes (CPU), ~5-15 minutes (GPU)
- **Model Size**: ~2-5MB
- **Memory Usage**: ~2-4GB during training
- **Inference Speed**: <1ms per image

## Advanced Features

### Data Augmentation
- **Random Rotation**: ±10 degrees
- **Random Zoom**: ±10% scaling
- **Random Translation**: ±10% shift
- **Real-time Augmentation**: Applied during training

### Training Callbacks
- **Early Stopping**: Prevents overfitting
- **Model Checkpointing**: Saves best model
- **Learning Rate Reduction**: Adaptive learning rate
- **Progress Monitoring**: Real-time training metrics

### Model Persistence
- **H5 Format**: Standard Keras model saving
- **Training History**: Complete training metrics
- **Metadata**: Timestamp and performance data

## File Structure
```
MNIST Handwritten Digit Recognition/
├── MNIST_CNN.py                    # Main application file
├── mnist_cnn_model.h5             # Trained model (auto-generated)
├── mnist_best_model.h5            # Best model checkpoint (auto-generated)
├── training_history.joblib        # Training history (auto-generated)
├── training_history.png           # Training curves plot (auto-generated)
├── confusion_matrix.png           # Confusion matrix plot (auto-generated)
├── sample_predictions.png         # Sample predictions plot (auto-generated)
└── README.md                      # This documentation
```

## Dependencies
- `tensorflow`: Deep learning framework
- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical data visualization
- `scikit-learn`: Machine learning utilities
- `joblib`: Model serialization

## Configuration Options

### Model Parameters
```python
input_shape = (28, 28, 1)          # Input image dimensions
num_classes = 10                    # Number of digit classes
epochs = 50                         # Maximum training epochs
batch_size = 128                    # Training batch size
val_split = 0.2                     # Validation data split
```

### Data Augmentation
```python
rotation_range = 0.1                # Rotation range
zoom_range = 0.1                    # Zoom range
translation_range = 0.1             # Translation range
```

### Callback Parameters
```python
early_stopping_patience = 10        # Early stopping patience
reduce_lr_patience = 5              # Learning rate reduction patience
min_lr = 0.0001                     # Minimum learning rate
```

## Troubleshooting

### Common Issues
1. **Memory Issues**
   - Reduce batch size
   - Close unnecessary applications
   - Use data generators for large datasets

2. **Slow Training**
   - Enable GPU acceleration
   - Reduce model complexity
   - Use mixed precision training

3. **Poor Performance**
   - Increase training epochs
   - Adjust learning rate
   - Modify data augmentation
   - Check data preprocessing

4. **Model Not Saving**
   - Check file permissions
   - Ensure sufficient disk space
   - Verify model compilation

### Error Messages
- `CUDA out of memory`: Reduce batch size or use CPU
- `Model compilation failed`: Check layer compatibility
- `Data loading error`: Verify MNIST dataset availability

## Example Usage

### Training a New Model
```python
from MNIST_CNN import main
main()  # Run complete training pipeline
```

### Loading Pre-trained Model
```python
from MNIST_CNN import load_trained_model, predict_digit
import numpy as np

# Load model
model = load_trained_model('mnist_cnn_model.h5')

# Predict single image
image = np.random.rand(28, 28)  # Your image data
digit, confidence = predict_digit(model, image)
print(f"Predicted digit: {digit}, Confidence: {confidence:.2f}")
```

### Custom Training
```python
from MNIST_CNN import load_and_preprocess_data, build_cnn_model, train_model

# Load data
(x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

# Build model
model = build_cnn_model()

# Train with custom parameters
history = train_model(model, x_train, y_train, x_test, y_test, 
                     epochs=30, batch_size=64)
```

## Performance Optimization

### GPU Acceleration
```python
# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### Mixed Precision Training
```python
# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
**Om Ravindra Wakhare**  
Date: 18/09/2025

## Version History
- **v1.0.0** (18/09/2025): Initial release with complete CNN implementation

## Support
For issues and questions, please create an issue in the repository or contact the author.

## Future Enhancements
- [ ] Support for custom digit datasets
- [ ] Real-time webcam digit recognition
- [ ] Model quantization for mobile deployment
- [ ] Advanced data augmentation techniques
- [ ] Hyperparameter optimization
- [ ] Ensemble methods
- [ ] Transfer learning capabilities
- [ ] Web interface for predictions
- [ ] API endpoint for model serving
- [ ] Docker containerization
