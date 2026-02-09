# Cable Defect Detection Model

## Overview

**Model Name:** Defect Detection of Cable Data using Autoencoder and Pseudo Defect Generation  
**Type:** Detection  
**Accuracy:** *7% - Trained with two-stage detection system  

This model performs automated defect detection on cable sensor images using a combination of traditional computer vision and deep learning techniques.

## How It Works

The model uses a **two-stage detection pipeline**:

### Stage 1: Cable Extraction
- Automatically extracts individual cables from multi-cable sensor images
- Uses contour detection to identify and separate cables
- Handles images with multiple cables (typically 64+ cables per image)

### Stage 2: Defect Classification
Each extracted cable is classified using two methods:

1. **Cut Detection** (Hough Line Transform)
   - Detects diagonal and vertical cuts (≥30° from horizontal)
   - Uses edge detection and line detection algorithms
   - Immediately classifies as "Defective" if cuts are found

2. **Autoencoder-based Detection** (Deep Learning)
   - Trained on patches from non-defective cables
   - Detects anomalies including:
     - Large dark spots
     - Burn areas
     - Other surface defects
   - Uses reconstruction error to identify defects

## Outputs

For each input image, the model returns:

- **Classification:** Overall status (e.g., "3 Defective cable(s) detected" or "All cables are Good")
- **Per-Cable Results:** Individual classification for each extracted cable
- **Visualization:** Annotated image showing all cables with color-coded bounding boxes
  - Green = Good
  - Red = Defective
- **Detailed Attributes:**
  - Total Cables
  - Defective Count
  - Good Count
  - Cut Detections
  - Autoencoder Detections
  - Reconstruction Error Statistics
  - Threshold Used

## Installation

### Requirements

```bash
pip install tensorflow opencv-python numpy
```

### Required Files

1. **CableDefectDetection.py** - Main model class
2. **sensor_autoencoder.keras** - Trained autoencoder weights

### GPU Support (Optional but Recommended)

The model automatically uses GPU if TensorFlow-GPU is installed and a compatible GPU is available.

## Usage

### Basic Usage

```python
from CableDefectDetection import CableDefectDetection

# Initialize model
model = CableDefectDetection(
    results_dir='./results',
    model_path='sensor_autoencoder.keras'
)

# Predict on an image
results = model.predict(
    image_path='path/to/sensor_image.jpg',
    run_id='test_001'
)

# Access results
print(results['output'])  # e.g., "2 Defective cable(s) detected"
print(results['attributes'])  # Dictionary of metrics
```

### Advanced Configuration

```python
# Initialize with custom parameters
model = CableDefectDetection(
    results_dir='./results',
    model_path='sensor_autoencoder.keras',
    patch_size=64,              # Patch size for autoencoder (default: 64)
    stride=32,                  # Stride for patch extraction (default: 32)
    min_cut_angle=30,           # Minimum angle for cut detection (default: 30)
    threshold=0.0035,           # Reconstruction error threshold (default: 0.0035)
    min_sensor_width_ratio=0.3  # Minimum cable width ratio (default: 0.3)
)
```

### Return Value Structure

```python
{
    'output': 'Classification summary string',
    'attributes': {
        'Total Cables': 64,
        'Defective': 3,
        'Good': 61,
        'Cut Detections': 1,
        'Autoencoder Detections': 2,
        'Avg Reconstruction Error': 0.002145,
        'Max Reconstruction Error': 0.008234,
        'Threshold': 0.0035
    },
    'visualization_filename': 'test_001_visualization.jpg',
    'image_info': {
        'filename': 'sensor_image.jpg',
        'width': 2448,
        'height': 2048
    }
}
```

## Output Files

The model generates the following files in `results_dir`:

1. **{run_id}_visualization.jpg** - Annotated image with bounding boxes and labels
2. **{run_id}_results.json** - Detailed JSON results with per-cable classifications

## Training Details

### Dataset
- **Non-defective samples:** Normal cables with small surface spots (considered normal)
- **Defective samples:** Cables with:
  - Diagonal/vertical cuts
  - Large dark spots
  - Burn areas
  - Other significant defects

### Preprocessing
- Minimal preprocessing to preserve defect features
- Patch-based approach (64×64 patches with stride 32)
- Grayscale conversion only

### Augmentation
- Pseudo-defect generation during training:
  - Large dark spots
  - Diagonal cracks (30-90° angles)
  - Vertical cracks
  - Burn areas

### Model Architecture
- **Encoder:** 3 convolutional layers (16→32→64 filters)
- **Latent Space:** 128-dimensional dense layer
- **Decoder:** 3 transpose convolutional layers (64→32→16→8 filters)
- **Total Parameters:** ~150K (lightweight for fast inference)

### Training Parameters
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam (learning rate: 0.001)
- **Epochs:** 30 (with early stopping)
- **Batch Size:** 64

## Threshold Calibration

The default threshold (0.0035) was determined by:
1. Computing reconstruction errors on non-defective validation set
2. Setting threshold at mean + 2×std deviation
3. Fine-tuning based on validation results

**To recalibrate for your data:**
```python
model.threshold = 0.004  # Increase to reduce false positives
```

## Performance Characteristics

- **Inference Speed:** ~0.5-2 seconds per sensor image (depends on number of cables)
- **Cut Detection:** Very fast (milliseconds per cable)
- **Autoencoder:** Moderate (depends on patch count)
- **Scalability:** Handles 64+ cables per image efficiently

## Known Limitations

1. **Small spots are normal:** The model is trained to accept small surface spots as normal features
2. **Horizontal lines:** May not detect horizontal defects as effectively as diagonal/vertical
3. **Image quality:** Requires clear, well-lit sensor images
4. **Cable separation:** May struggle if cables overlap significantly

## Troubleshooting

### No cables detected
- Check image quality and lighting
- Verify image contains clear cable boundaries
- Adjust `min_sensor_width_ratio` parameter

### Too many false positives
- Increase `threshold` parameter
- Review if "defects" are actually normal variations

### Too many false negatives
- Decrease `threshold` parameter
- Check if defects are similar to training data

## Citation

If you use this model, please cite:

```
Cable Defect Detection using Autoencoder and Pseudo Defect Generation
Two-stage detection system with cut detection and deep learning
```

## License

[Specify your license here]

## Contact

[Your contact information here]

## Model File Information

**Required file:** `sensor_autoencoder.keras`

This file contains the trained weights for the autoencoder model. It should be placed in the same directory as the Python script or specify the path when initializing the model.

**Model format:** TensorFlow/Keras SavedModel (.keras)
**File size:** ~1-2 MB (lightweight model)

To obtain the model weights file:
1. If you have trained the model, it will be saved as `sensor_autoencoder.keras`
2. Contact the model author for pre-trained weights
3. See training code in original implementation for retraining

## Version History

- **v1.0** - Initial release with two-stage detection system
  - Cable extraction from multi-cable images
  - Cut detection using Hough transform
  - Autoencoder-based anomaly detection
  - Comprehensive visualization
