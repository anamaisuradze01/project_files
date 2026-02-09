# Cable Measurement Model (with Image Cutting)

## Model Information

**Model Name:** Cable Measurements

**Type:** Measurement

**Accuracy:** 96%

**Note on Accuracy:** If a cable has a defect on the border, the model classifies it as part of the border and calculates the thickness of the cable with that in mind. This design choice ensures robust measurements even with border defects.

## Features

This model has two modes of operation:

1. **Direct Measurement Mode** (`cut_image=False`): Process the entire image as-is
2. **Cut and Measure Mode** (`cut_image=True`): Automatically cut the image into parts before measuring each part

### Image Cutting

When enabled, the model cuts each input image into a grid:
- **Vertical splits** (columns): Default 2 (configurable via `num_columns`)
- **Horizontal splits** (rows): Default 15 (configurable via `num_rows`)

This is useful for processing large images with multiple cable strips, where each strip needs individual measurement.

## Model Attributes

### When `cut_image=False` (Direct Mode):

**Combined (All Cables) Measurements:**
- **Cables Detected** - Number of cables found in the image
- **Combined Mean Thickness (px)** - Average thickness across all cables
- **Combined Std Dev (px)** - Standard deviation of thickness across all cables
- **Combined Min Thickness (px)** - Minimum thickness found across all cables
- **Combined Max Thickness (px)** - Maximum thickness found across all cables
- **Combined CV (%)** - Coefficient of variation (percentage)
- **Total Data Points** - Total number of thickness measurements taken

**Individual Cable Measurements:**
For each detected cable (Cable 1, Cable 2, etc.):
- **Cable X Avg (px)** - Average thickness for that specific cable
- **Cable X Std Dev (px)** - Standard deviation for that specific cable
- **Cable X Min (px)** - Minimum thickness for that specific cable
- **Cable X Max (px)** - Maximum thickness for that specific cable

### When `cut_image=True` (Cut and Measure Mode):

**Summary:**
- **Total Parts** - Number of parts the image was cut into
- **Total Cables Detected** - Total cables across all parts

**Per-Part Measurements:**
For each part (e.g., `run_id_c0_r00`, `run_id_c1_r00`, etc.):
- **Part_Name - Cables Detected** - Number of cables in this part
- **Part_Name - Combined Mean (px)** - Average thickness in this part
- **Part_Name - Combined Std Dev (px)** - Standard deviation in this part
- **Part_Name - Combined Min (px)** - Minimum thickness in this part
- **Part_Name - Combined Max (px)** - Maximum thickness in this part
- **Part_Name - Combined CV (%)** - Coefficient of variation in this part

## How It Works

The model uses computer vision techniques to:

1. **Optional: Cut Image** (if enabled):
   - Divides image into column × row grid
   - Each part is processed independently
   
2. **Detect Cables**: Identifies cable contours using binary thresholding and morphological operations

3. **Extract Boundaries**: Finds the upper and lower boundaries of each cable

4. **Measure Thickness**: Calculates vertical thickness at each horizontal position along the cable

5. **Statistical Analysis**: Computes mean, standard deviation, min, max, and coefficient of variation

6. **Visualization**: Creates annotated images showing:
   - Green contours around detected cables
   - Magenta arrows marking minimum thickness points
   - Cyan arrows marking maximum thickness points
   - Blue arrows marking average thickness points with standard deviation
   - Orange arrows showing distances between cables

## Dependencies

- opencv-python (cv2)
- numpy
- scipy
- Pillow (PIL)

## Model Weights

This model does not require pre-trained weights as it uses classical computer vision algorithms.

## Usage Examples

### Example 1: Direct Measurement (No Cutting)

```python
from cable_measurement_model_with_cutting import CableMeasurement

# Initialize model without cutting
model = CableMeasurement(results_dir="./results", cut_image=False)

# Run prediction
result = model.predict(image_path="cable_image.jpg", run_id="test_001")

print(f"Output: {result['output']}")
print(f"Cables detected: {result['attributes']['Cables Detected']}")
```

### Example 2: Cut and Measure Mode

```python
from cable_measurement_model_with_cutting import CableMeasurement

# Initialize model with cutting enabled
model = CableMeasurement(
    results_dir="./results", 
    cut_image=True,
    num_columns=2,  # 2 vertical splits
    num_rows=15     # 15 horizontal splits
)

# Run prediction - image will be automatically cut into 30 parts (2×15)
result = model.predict(image_path="large_cable_sheet.jpg", run_id="sheet_001")

print(f"Output: {result['output']}")
print(f"Total parts: {result['attributes']['Total Parts']}")
print(f"Total cables: {result['attributes']['Total Cables Detected']}")
```

### Example 3: Custom Grid Size

```python
# Custom cutting configuration
model = CableMeasurement(
    results_dir="./results", 
    cut_image=True,
    num_columns=3,  # 3 vertical splits
    num_rows=10     # 10 horizontal splits
)

result = model.predict(image_path="image.jpg", run_id="custom_001")
```

## Output Format

The model returns a dictionary with:
- `output`: Status message
- `attributes`: Dictionary of measurement metrics (varies by mode)
- `visualization_filename`: Filename of the main annotated output image
- `image_info`: Original image metadata (filename, width, height)

### Output Files

**Direct Mode:**
- `{run_id}_vis.jpg` - Single visualization image with measurements

**Cut and Measure Mode:**
- `{run_id}_c{col}_r{row}_vis.jpg` - Visualization for each part that contains cables
- Multiple files, one per part with detected cables

## Notes

- In cut mode, only parts that contain detected cables will have visualizations saved
- The cutting algorithm distributes any extra pixels from division evenly across the first rows
- Border pixels (5px on each side) are skipped during thickness measurement to avoid edge artifacts
