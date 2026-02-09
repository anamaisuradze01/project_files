# Cable Measurement Model

## Model Information

**Model Name:** Cable Measurements

**Type:** Measurement

**Accuracy:** 96%

**Note on Accuracy:** If a cable has a defect on the border, the model classifies it as part of the border and calculates the thickness of the cable with that in mind. This design choice ensures robust measurements even with border defects.

## Model Attributes

The model returns the following attributes for each analyzed image:

### Combined (All Cables) Measurements:
- **Cables Detected** - Number of cables found in the image
- **Combined Mean Thickness (px)** - Average thickness across all cables
- **Combined Std Dev (px)** - Standard deviation of thickness across all cables
- **Combined Min Thickness (px)** - Minimum thickness found across all cables
- **Combined Max Thickness (px)** - Maximum thickness found across all cables
- **Combined CV (%)** - Coefficient of variation (percentage)
- **Total Data Points** - Total number of thickness measurements taken

### Individual Cable Measurements:
For each detected cable (Cable 1, Cable 2, etc.):
- **Cable X Avg (px)** - Average thickness for that specific cable
- **Cable X Std Dev (px)** - Standard deviation for that specific cable
- **Cable X Min (px)** - Minimum thickness for that specific cable
- **Cable X Max (px)** - Maximum thickness for that specific cable

## How It Works

The model uses computer vision techniques to:

1. **Detect Cables**: Identifies cable contours in the image using binary thresholding and morphological operations
2. **Extract Boundaries**: Finds the upper and lower boundaries of each cable
3. **Measure Thickness**: Calculates vertical thickness at each horizontal position along the cable
4. **Statistical Analysis**: Computes mean, standard deviation, min, max, and coefficient of variation
5. **Visualization**: Creates annotated images showing:
   - Green contours around detected cables
   - Magenta arrows marking minimum thickness points
   - Cyan arrows marking maximum thickness points
   - Blue arrows marking average thickness points with standard deviation
   - Orange arrows showing distances between cables

## Dependencies

- opencv-python (cv2)
- numpy
- scipy

## Model Weights

This model does not require pre-trained weights as it uses classical computer vision algorithms (contour detection, morphological operations, and statistical analysis).

## Usage Example

```python
from cable_measurement_model import CableMeasurement

# Initialize model
model = CableMeasurement(results_dir="./results")

# Run prediction
result = model.predict(image_path="cable_image.jpg", run_id="test_001")

print(f"Output: {result['output']}")
print(f"Attributes: {result['attributes']}")
print(f"Visualization saved as: {result['visualization_filename']}")
```

## Output Format

The model returns a dictionary with:
- `output`: Status message (e.g., "5 cables detected - Measurements completed")
- `attributes`: Dictionary of measurement metrics
- `visualization_filename`: Filename of the annotated output image
- `image_info`: Original image metadata (filename, width, height)
