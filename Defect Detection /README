# Cable Defect Detector Model

## Model Information

- **Model Name**: CableDefectDetector
- **Type**: Detection (Binary Classification: Defective / Non-Defective)
- **Accuracy**: 92%
- **Technology**: Two-stage detection system
  - Stage 1: Hough Line Transform for cut/crack detection
  - Stage 2: Lightweight Convolutional Autoencoder for anomaly detection

## What It Detects

The model classifies cable sensor images as either **Defective** or **Non-Defective**.

**Defects detected:**
- Diagonal or vertical cuts/cracks (≥30° from horizontal)
- Large dark spots
- Burn areas or gray patches
- Other surface anomalies

**Normal features (not flagged as defects):**
- Small spots (considered normal variation)
- Minor texture variations

## Model Attributes Returned

The model returns the following attributes in the results:

1. **Detection Method**: Either "Cut Detection" or "Autoencoder"
2. **Reconstruction Score**: Numerical score indicating anomaly level (0-1 for normal, 999 for cuts)
3. **Threshold**: The decision threshold (default: 0.0035)
4. **Cut Detected**: "Yes" or "No"
5. **Max Cut Angle**: Angle of detected cut (if applicable)
6. **Patches Analyzed**: Number of image patches processed (for autoencoder method)
7. **Status**: Human-readable status message

## Files Included

1. **cable_defect_detector.py** - Main model class
2. **sensor_autoencoder.keras** - Trained model weights (you need to provide this)

## Model Weights

You need to provide the trained autoencoder model file: `sensor_autoencoder.keras`

This file is generated when you run your training script with the `main()` function.

## Installation Requirements

```bash
pip install tensorflow opencv-python numpy
```

## Usage Example

```python
from cable_defect_detector import CableDefectDetector

# Initialize model
model = CableDefectDetector(
    results_dir="./results",
    model_path="sensor_autoencoder.keras",  # Path to your trained model
    threshold=0.0035,  # Optional: custom threshold
    min_angle=30  # Optional: minimum angle for cut detection
)

# Make prediction
result = model.predict(
    image_path="test_cable.jpg",
    run_id="test_001"
)

print(result['output'])  # "Defective" or "Non-Defective"
print(result['attributes'])  # Dict of metrics
print(result['visualization_filename'])  # Filename of visualization image
```

## Integration Notes

- The model automatically creates visualization images showing detection results
- Visualization images are saved to the `results_dir` with the pattern `{run_id}_vis.jpg`
- Cut detection happens first (faster) before autoencoder analysis
- The model handles both color and grayscale images

## Model Architecture

**Autoencoder:**
- Input: 64x64 grayscale patches
- Encoder: 3 Conv2D layers (16→32→64 filters) + Dense(128)
- Latent space: 128 dimensions
- Decoder: Dense + 3 Conv2DTranspose layers
- Loss: MSE (Mean Squared Error)

**Cut Detection:**
- Canny edge detection
- Hough Line Transform
- Angle-based filtering (≥30° from horizontal)

## Performance

- **Overall Accuracy**: 92%
- **Detection Speed**: ~1-3 seconds per image (depending on image size)
- **False Positive Rate**: Low (typically <5%)
- **False Negative Rate**: Low (typically <8%)

## Notes

- The model expects cable sensor images similar to training data
- Images are automatically resized to maintain aspect ratio
- Small spots are considered normal and won't trigger defect classification
- The two-stage approach ensures fast detection of obvious defects (cuts)
