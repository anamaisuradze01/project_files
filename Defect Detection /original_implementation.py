import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import random
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# ====================================
# CUT DETECTION - PRE-SCREENING
# ====================================
def preprocess_for_cut_detection(img):
    """
    Special preprocessing for detecting cuts/lines.
    Different from autoencoder preprocessing.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding or edge detection
    # Using Canny edge detection for line detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    return edges


def detect_cuts_hough(img, min_angle=30):
    """
    Detect diagonal or vertical cuts using Hough Line Transform.

    Args:
        img: Input image (BGR or grayscale)
        min_angle: Minimum angle from horizontal (30-90 degrees for diagonal/vertical)

    Returns:
        bool: True if diagonal/vertical cuts detected, False otherwise
        list: List of detected line angles (for debugging)
    """
    # Preprocess for cut detection
    edges = preprocess_for_cut_detection(img)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,  # Minimum votes to consider a line
        minLineLength=50,  # Minimum line length
        maxLineGap=10  # Maximum gap between line segments
    )

    if lines is None:
        return False, []

    detected_angles = []
    cut_detected = False

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate angle from horizontal
        if x2 - x1 == 0:
            # Vertical line
            angle = 90
        else:
            angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))

        detected_angles.append(angle)

        # Check if line is diagonal (30-60 degrees) or vertical (60-90 degrees)
        if angle >= min_angle:
            cut_detected = True

    return cut_detected, detected_angles


def visualize_cut_detection(sensor_path, min_angle=30):
    """Visualize cut detection process"""
    img = cv2.imread(sensor_path)
    if img is None:
        print(f"Could not load {sensor_path}")
        return

    # Preprocessing
    edges = preprocess_for_cut_detection(img)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=10)

    # Draw detected lines
    img_with_lines = img.copy()
    cut_detected = False

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle
            if x2 - x1 == 0:
                angle = 90
            else:
                angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))

            # Color code: red for cuts (>30°), green for normal lines
            if angle >= min_angle:
                color = (0, 0, 255)  # Red for cuts
                cut_detected = True
            else:
                color = (0, 255, 0)  # Green for normal

            cv2.line(img_with_lines, (x1, y1), (x2, y2), color, 2)

    # Display
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Edge Detection (Canny)', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
    title = f'Detected Lines (Red=Cut ≥{min_angle}°, Green=Normal)'
    if cut_detected:
        title += '\n⚠️ CUT DETECTED - DEFECTIVE'
        axes[2].set_title(title, fontsize=12, color='red', weight='bold')
    else:
        title += '\n✓ No Cuts - Pass to Autoencoder'
        axes[2].set_title(title, fontsize=12, color='green', weight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# ====================================
# PREPROCESSING - MINIMAL TO PRESERVE DEFECTS (for Autoencoder)
# ====================================
def preprocess_sensor(img):
    """
    Minimal preprocessing - preserve ALL features including small spots.
    Small spots are NORMAL and should be learned by autoencoder.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # NO morphological operations - they remove defects
    # NO blurring - it smooths out defects
    # Just return the grayscale image as-is

    return img


def resize_sensor(img, target_height=128):
    """Resize sensor to standard height while maintaining aspect ratio"""
    h, w = img.shape[:2]
    if h == 0:
        return img
    aspect_ratio = w / h
    target_width = int(target_height * aspect_ratio)
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)


# ====================================
# DEFECT GENERATION - Only for augmentation
# ====================================
def add_large_dark_spots(img, num_spots=None):
    """Add LARGE dark spot defects (significantly bigger than normal spots)"""
    img_copy = img.copy().astype(np.float32)
    h, w = img_copy.shape[:2]

    if num_spots is None:
        num_spots = random.randint(1, 3)

    for _ in range(num_spots):
        cx = random.randint(int(w * 0.15), int(w * 0.85))
        cy = random.randint(int(h * 0.15), int(h * 0.85))

        # Very large spots (20-50 pixels radius)
        radius = random.randint(20, 50)

        # Very dark
        color = random.randint(0, 70)

        cv2.circle(img_copy, (cx, cy), radius, float(color), -1)

        # Blur edges
        mask = np.zeros_like(img_copy)
        cv2.circle(mask, (cx, cy), radius + 6, 255, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 0) / 255.0
        img_copy = img_copy * (1 - mask * 0.6) + color * mask * 0.6

    return img_copy.astype(np.uint8)


def add_diagonal_crack(img, num_cracks=1, min_angle=30):
    """
    Add diagonal crack/cut defects at 30+ degrees from horizontal.

    Args:
        img: Input image
        num_cracks: Number of cracks to add
        min_angle: Minimum angle from horizontal (default 30 degrees)
    """
    img_copy = img.copy().astype(np.float32)
    h, w = img_copy.shape[:2]

    for _ in range(num_cracks):
        # Generate random angle between min_angle and 90 degrees
        angle_deg = random.uniform(min_angle, 85)
        angle_rad = np.radians(angle_deg)

        # Random starting point
        x1 = random.randint(int(w * 0.1), int(w * 0.9))
        y1 = random.randint(int(h * 0.1), int(h * 0.4))

        # Calculate end point based on angle
        # Length of crack
        crack_length = random.randint(int(h * 0.4), int(h * 0.7))

        x2 = int(x1 + crack_length * np.cos(angle_rad))
        y2 = int(y1 + crack_length * np.sin(angle_rad))

        # Ensure points are within bounds
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        thickness = random.randint(2, 5)
        color = random.randint(0, 60)

        cv2.line(img_copy, (x1, y1), (x2, y2), float(color), thickness)

        # Add blur along line
        mask = np.zeros_like(img_copy)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness + 6)
        mask = cv2.GaussianBlur(mask, (7, 7), 0) / 255.0
        img_copy = img_copy * (1 - mask * 0.4) + color * mask * 0.4

    return img_copy.astype(np.uint8)


def add_vertical_crack(img, num_cracks=1):
    """Add vertical crack/cut defects (80-90 degrees from horizontal)"""
    img_copy = img.copy().astype(np.float32)
    h, w = img_copy.shape[:2]

    for _ in range(num_cracks):
        # Add slight randomness to vertical (80-90 degrees)
        angle_deg = random.uniform(80, 90)

        x = random.randint(int(w * 0.2), int(w * 0.8))
        y_start = random.randint(0, int(h * 0.3))

        # Calculate length
        crack_length = random.randint(int(h * 0.4), int(h * 0.7))

        # Small horizontal deviation for realism
        x_end = x + random.randint(-5, 5)
        y_end = y_start + crack_length
        y_end = min(h - 1, y_end)

        thickness = random.randint(2, 5)
        color = random.randint(0, 60)

        cv2.line(img_copy, (x, y_start), (x_end, y_end), float(color), thickness)

        mask = np.zeros_like(img_copy)
        cv2.line(mask, (x, y_start), (x_end, y_end), 255, thickness + 6)
        mask = cv2.GaussianBlur(mask, (7, 7), 0) / 255.0
        img_copy = img_copy * (1 - mask * 0.4) + color * mask * 0.4

    return img_copy.astype(np.uint8)


def add_burn_area(img, num_burns=1):
    """Add burn/gray area defects (irregular gray patches)"""
    img_copy = img.copy().astype(np.float32)
    h, w = img_copy.shape[:2]

    for _ in range(num_burns):
        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))

        # Irregular shape using ellipse with random rotation
        axes_x = random.randint(25, 60)
        axes_y = random.randint(20, 50)
        angle = random.randint(0, 180)

        # Gray/dark color for burn
        color = random.randint(60, 130)

        cv2.ellipse(img_copy, (cx, cy), (axes_x, axes_y), angle, 0, 360, float(color), -1)

        # Heavy blur for burn effect
        mask = np.zeros_like(img_copy)
        cv2.ellipse(mask, (cx, cy), (axes_x + 10, axes_y + 10), angle, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0) / 255.0
        img_copy = img_copy * (1 - mask * 0.7) + color * mask * 0.7

    return img_copy.astype(np.uint8)


def generate_realistic_defect(img):
    """Generate realistic defects: large spots, cracks (30+ degrees), curly lines, burns"""
    defect_type = random.choice(['large_spots', 'diagonal_crack', 'vertical_crack',
                                 'burn', 'mixed'])

    if defect_type == 'large_spots':
        return add_large_dark_spots(img, num_spots=random.randint(2, 4))
    elif defect_type == 'diagonal_crack':
        # Use min_angle=30 to ensure diagonal cuts are 30+ degrees
        return add_diagonal_crack(img, num_cracks=random.randint(1, 2), min_angle=30)
    elif defect_type == 'vertical_crack':
        return add_vertical_crack(img, num_cracks=1)
    elif defect_type == 'burn':
        return add_burn_area(img, num_burns=random.randint(1, 2))
    else:  # mixed - combine multiple defect types
        img = add_large_dark_spots(img, num_spots=1)
        if random.random() > 0.5:
            img = add_diagonal_crack(img, num_cracks=1, min_angle=30)
        return img


# ====================================
# PATCH HANDLING
# ====================================
def extract_patches(img, patch_size=64, stride=32):
    """Extract overlapping patches from sensor image"""
    patches = []
    h, w = img.shape[:2]

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y:y + patch_size, x:x + patch_size]
            # Lower threshold - include more patches with small variations
            if patch.std() > 2:
                patches.append(patch)

    return patches


def patches_to_numpy(patches):
    """Convert patches to numpy array"""
    if len(patches) == 0:
        return np.array([]).reshape(0, 64, 64, 1)
    arr = np.stack(patches).astype("float32") / 255.0
    arr = arr[..., np.newaxis]
    return arr


# ====================================
# LIGHTWEIGHT AUTOENCODER
# ====================================
def build_lightweight_autoencoder(input_shape=(64, 64, 1)):
    """Ultra-lightweight autoencoder optimized for speed"""
    inp = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(16, 3, activation='relu', padding='same', strides=2)(inp)  # 32x32
    x = layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(x)  # 16x16
    x = layers.Conv2D(64, 3, activation='relu', padding='same', strides=2)(x)  # 8x8

    x = layers.Flatten()(x)
    latent = layers.Dense(128, activation='relu')(x)

    # Decoder
    x = layers.Dense(8 * 8 * 64, activation='relu')(latent)
    x = layers.Reshape((8, 8, 64))(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', padding='same', strides=2)(x)  # 16x16
    x = layers.Conv2DTranspose(16, 3, activation='relu', padding='same', strides=2)(x)  # 32x32
    x = layers.Conv2DTranspose(8, 3, activation='relu', padding='same', strides=2)(x)  # 64x64

    out = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

    return model


# ====================================
# TRAINING
# ====================================
def train_autoencoder(model, train_patches, val_patches=None, epochs=30, batch_size=64):
    """Train autoencoder with early stopping"""
    callbacks = [
        EarlyStopping(monitor='val_loss' if val_patches is not None else 'loss',
                      patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss' if val_patches is not None else 'loss',
                          factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        train_patches, train_patches,
        validation_data=(val_patches, val_patches) if val_patches is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


# ====================================
# SENSOR SCORING WITH CUT PRE-SCREENING
# ====================================
def compute_sensor_score_with_cut_detection(model, sensor_img_path, patch_size=64, stride=32,
                                            min_angle=30, cut_detection_enabled=True):
    """
    Two-stage classification:
    1. Check for diagonal/vertical cuts (≥30 degrees)
    2. If no cuts, use autoencoder for other defects

    Args:
        model: Trained autoencoder model
        sensor_img_path: Path to sensor image
        patch_size: Patch size for autoencoder
        stride: Stride for patch extraction
        min_angle: Minimum angle for cut detection (default 30 degrees)
        cut_detection_enabled: Whether to use cut pre-screening

    Returns:
        tuple: (score, detection_method, cut_detected)
    """
    img = cv2.imread(sensor_img_path)
    if img is None:
        return 0.0, "error", False

    # STAGE 1: Cut Detection
    if cut_detection_enabled:
        cut_detected, angles = detect_cuts_hough(img, min_angle=min_angle)

        if cut_detected:
            # If cut detected, mark as defective immediately
            # Return high score to indicate defect
            return 999.0, "cut_detection", True

    # STAGE 2: Autoencoder for other defects (spots, burns)
    # Use different preprocessing for autoencoder
    preprocessed = preprocess_sensor(img)
    resized = resize_sensor(preprocessed)
    patches = extract_patches(resized, patch_size, stride)

    if len(patches) == 0:
        return 0.0, "autoencoder", False

    X = patches_to_numpy(patches)
    recon = model.predict(X, batch_size=128, verbose=0)
    errors = np.mean(np.square(recon - X), axis=(1, 2, 3))

    # Use maximum error - focuses on the worst patch (where defect is)
    return np.max(errors), "autoencoder", False


# ====================================
# DATA COLLECTION
# ====================================
def collect_training_patches(sensor_dir, patch_size=64, stride=32,
                             max_patches_per_sensor=80, max_sensors=None,
                             augment_with_pseudo_defects=True,
                             cut_detection_enabled=True):
    """
    Collect patches from NON-DEFECTIVE sensors (after cut pre-screening).
    These contain small spots which are NORMAL.

    Optionally augment with pseudo-defects to help model learn
    what abnormal looks like during training.
    """
    all_patches = []
    sensor_files = glob.glob(os.path.join(sensor_dir, "*"))

    if max_sensors and len(sensor_files) > max_sensors:
        random.shuffle(sensor_files)
        sensor_files = sensor_files[:max_sensors]

    print(f"Collecting patches from {len(sensor_files)} non-defective sensors...")
    print("Note: Sensors with diagonal/vertical cuts (≥30°) will be filtered out during training.")
    print("Small spots in remaining sensors are NORMAL and should be learned.")

    sensors_with_cuts = 0

    for sensor_path in tqdm(sensor_files, desc="Processing sensors"):
        img = cv2.imread(sensor_path)
        if img is None:
            continue

        # Pre-screen for cuts if enabled
        if cut_detection_enabled:
            cut_detected, _ = detect_cuts_hough(img, min_angle=30)
            if cut_detected:
                sensors_with_cuts += 1
                continue  # Skip this sensor, it has cuts

        # Process as-is - small spots are NORMAL
        preprocessed = preprocess_sensor(img)
        resized = resize_sensor(preprocessed)
        patches = extract_patches(resized, patch_size, stride)

        if len(patches) > max_patches_per_sensor:
            patches = random.sample(patches, max_patches_per_sensor)

        all_patches.extend(patches)

        # Add pseudo-defects for better separation (30% of images)
        # This helps model learn what is NOT normal (spots, burns, but NOT cuts)
        if augment_with_pseudo_defects and random.random() > 0.7:
            for _ in range(random.randint(1, 2)):
                # Generate non-cut defects (spots and burns only)
                defect_type = random.choice(['large_spots', 'burn'])
                if defect_type == 'large_spots':
                    defective = add_large_dark_spots(resized.copy(), num_spots=random.randint(2, 4))
                else:
                    defective = add_burn_area(resized.copy(), num_burns=random.randint(1, 2))

                def_patches = extract_patches(defective, patch_size, stride)
                if len(def_patches) > max_patches_per_sensor // 2:
                    def_patches = random.sample(def_patches, max_patches_per_sensor // 2)
                all_patches.extend(def_patches)

    print(f"\nSensors filtered out due to cuts: {sensors_with_cuts}")
    print(f"Total patches collected: {len(all_patches)}")
    print("Small spots in these patches are considered NORMAL.")

    return patches_to_numpy(all_patches)


# ====================================
# VISUALIZATION
# ====================================
def visualize_preprocessing(sensor_path):
    """Show preprocessing steps and pseudo-defect examples"""
    img = cv2.imread(sensor_path)
    if img is None:
        print(f"Could not load {sensor_path}")
        return

    # Cut detection preprocessing
    edges = preprocess_for_cut_detection(img)

    # Autoencoder preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    preprocessed = preprocess_sensor(img)
    resized = resize_sensor(preprocessed)

    # Generate different types of defects
    pseudo_spots = add_large_dark_spots(resized.copy(), num_spots=3)
    pseudo_crack = add_diagonal_crack(resized.copy(), num_cracks=2, min_angle=30)
    pseudo_burn = add_burn_area(resized.copy(), num_burns=2)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original (has small normal spots)', fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Cut Detection (Edges)', fontsize=10)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(preprocessed, cmap='gray')
    axes[0, 2].set_title('Autoencoder Preprocessing', fontsize=10)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(resized, cmap='gray')
    axes[0, 3].set_title('Resized - Ready for Training', fontsize=10)
    axes[0, 3].axis('off')

    axes[1, 0].imshow(pseudo_spots, cmap='gray')
    axes[1, 0].set_title('DEFECT: Large Dark Spots', fontsize=10, color='red')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pseudo_crack, cmap='gray')
    axes[1, 1].set_title('DEFECT: Diagonal Cuts (≥30°)', fontsize=10, color='red')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(pseudo_burn, cmap='gray')
    axes[1, 2].set_title('DEFECT: Burns/Gray Areas', fontsize=10, color='red')
    axes[1, 2].axis('off')

    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_results(results_df, nondef_dir, def_dir, num_samples=50):
    """Visualize classification results with sample sensors"""
    samples = results_df.sample(min(num_samples, len(results_df)))

    cols = 10
    rows = (len(samples) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten() if len(samples) > 1 else [axes]

    for idx, (_, row) in enumerate(samples.iterrows()):
        if idx >= len(axes):
            break

        # Load sensor image
        if row['true_label'] == 'nondefective':
            sensor_path = os.path.join(nondef_dir, row['sensor_filename'])
        else:
            sensor_path = os.path.join(def_dir, row['sensor_filename'])

        img = cv2.imread(sensor_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(img)

        axes[idx].axis('off')

        # Color-coded label with detection method
        correct_mark = "✓" if row['correct'] else "✗"
        method = row['detection_method']

        if row['pred_label'] == 'defective':
            if method == 'cut_detection':
                label_text = f"{correct_mark} DEF (CUT)\n{row['reconstruction_error']:.1f}"
            else:
                label_text = f"{correct_mark} DEF (AE)\n{row['reconstruction_error']:.4f}"
            color = 'red' if row['correct'] else 'orange'
        else:
            label_text = f"{correct_mark} OK\n{row['reconstruction_error']:.4f}"
            color = 'green' if row['correct'] else 'purple'

        axes[idx].set_title(label_text, fontsize=11, color=color, weight='bold')

    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def create_score_histogram(results_df, threshold):
    """Display score distribution histogram"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Separate scores by detection method
    def_ae = results_df[(results_df['true_label'] == 'defective') &
                        (results_df['detection_method'] == 'autoencoder')]['reconstruction_error']
    def_cut = results_df[(results_df['true_label'] == 'defective') &
                         (results_df['detection_method'] == 'cut_detection')]['reconstruction_error']
    nondef_scores = results_df[results_df['true_label'] == 'nondefective']['reconstruction_error']

    ax.hist(nondef_scores, bins=50, alpha=0.6, label='Non-defective (small spots OK)',
            color='green', edgecolor='black')
    ax.hist(def_ae, bins=50, alpha=0.6, label='Defective (spots/burns - Autoencoder)',
            color='orange', edgecolor='black')

    # Cut detections will be at score=999
    if len(def_cut) > 0:
        ax.axvline(999, color='red', linestyle=':', linewidth=3,
                   label=f'Defective (cuts) - {len(def_cut)} detections')

    ax.axvline(threshold, color='blue', linestyle='--', linewidth=2,
               label=f'Threshold: {threshold:.5f}')

    ax.set_xlabel('Reconstruction Error', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Reconstruction Errors (Two-Stage Detection)', fontsize=14, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ====================================
# MAIN PIPELINE
# ====================================
def main():
    # Configuration
    NONDEF_DIR = "nondefective"  # Folder with non-defective sensors (has small normal spots)
    DEF_DIR = "defective"  # Folder with defective sensors (large spots, cracks, burns, curly lines)
    MODEL_PATH = "sensor_autoencoder.keras"

    PATCH_SIZE = 64
    STRIDE = 32
    EPOCHS = 30
    BATCH_SIZE = 64
    MIN_CUT_ANGLE = 30  # Minimum angle for cut detection (30-90 degrees)

    # Step 0: Demo cut detection visualization
    print("=" * 60)
    print("STEP 0: CUT DETECTION DEMONSTRATION")
    print("=" * 60)
    print(f"Detecting diagonal/vertical cuts (≥{MIN_CUT_ANGLE}° from horizontal)")
    sample_sensors = glob.glob(os.path.join(NONDEF_DIR, "*"))[:1]
    if sample_sensors:
        visualize_cut_detection(sample_sensors[0], min_angle=MIN_CUT_ANGLE)

    # Step 1: Visualize preprocessing on sample sensors
    print("\n" + "=" * 60)
    print("STEP 1: VISUALIZING PREPROCESSING")
    print("=" * 60)
    print("Note: Small spots in non-defective sensors are NORMAL")
    print("Defects = large spots, burns, OR diagonal/vertical cuts (≥30°)")
    if sample_sensors:
        visualize_preprocessing(sample_sensors[0])

    # Step 2: Collect training patches from non-defective sensors
    print("\n" + "=" * 60)
    print("STEP 2: COLLECTING TRAINING PATCHES")
    print("=" * 60)
    print("Cut pre-screening enabled - sensors with cuts will be filtered out")

    X_train = collect_training_patches(NONDEF_DIR, PATCH_SIZE, STRIDE,
                                       max_patches_per_sensor=80,
                                       augment_with_pseudo_defects=True,
                                       cut_detection_enabled=True)

    n = X_train.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(0.9 * n)
    X_tr, X_val = X_train[indices[:split]], X_train[indices[split:]]

    print(f"\nTraining patches: {X_tr.shape[0]}")
    print(f"Validation patches: {X_val.shape[0]}")

    # Step 3: Build and train autoencoder
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING AUTOENCODER")
    print("=" * 60)
    print("Learning that small spots are NORMAL...")
    print("Autoencoder will detect: large spots, burns, and other non-cut defects")

    model = build_lightweight_autoencoder(input_shape=(PATCH_SIZE, PATCH_SIZE, 1))
    model.summary()

    model, _ = train_autoencoder(model, X_tr, X_val, EPOCHS, BATCH_SIZE)
    model.save(MODEL_PATH)
    print(f"\n✓ Model saved to {MODEL_PATH}")

    # Step 4: Compute threshold from non-defective sensors (without cuts)
    print("\n" + "=" * 60)
    print("STEP 4: COMPUTING THRESHOLD")
    print("=" * 60)

    nondef_files = glob.glob(os.path.join(NONDEF_DIR, "*"))
    nondef_scores = []

    for sensor_path in tqdm(nondef_files, desc="Computing non-defective scores"):
        score, method, cut = compute_sensor_score_with_cut_detection(
            model, sensor_path, PATCH_SIZE, STRIDE,
            min_angle=MIN_CUT_ANGLE, cut_detection_enabled=True
        )
        # Only use autoencoder scores for threshold (skip cut detections)
        if method == "autoencoder":
            nondef_scores.append(score)

    nondef_scores = np.array(nondef_scores)
    mean_score = np.mean(nondef_scores)
    std_score = np.std(nondef_scores)
    threshold = 0.0035

    print(f"\nNon-defective sensor scores (autoencoder only):")
    print(f"  Mean: {mean_score:.5f}")
    print(f"  Std: {std_score:.5f}")
    print(f"  Selected threshold: {threshold:.5f}")

    # Step 5: Evaluate all sensors with two-stage detection
    print("\n" + "=" * 60)
    print("STEP 5: TWO-STAGE EVALUATION")
    print("=" * 60)
    print("Stage 1: Cut detection (diagonal/vertical ≥30°)")
    print("Stage 2: Autoencoder (spots, burns)")

    all_sensors = [(f, "nondefective") for f in glob.glob(os.path.join(NONDEF_DIR, "*"))] + \
                  [(f, "defective") for f in glob.glob(os.path.join(DEF_DIR, "*"))]

    results = []
    for sensor_path, true_label in tqdm(all_sensors, desc="Evaluating sensors"):
        score, method, cut_detected = compute_sensor_score_with_cut_detection(
            model, sensor_path, PATCH_SIZE, STRIDE,
            min_angle=MIN_CUT_ANGLE, cut_detection_enabled=True
        )

        # Classification logic
        if method == "cut_detection":
            pred_label = "defective"
        else:
            pred_label = "defective" if score > threshold else "nondefective"

        results.append({
            "sensor_filename": os.path.basename(sensor_path),
            "true_label": true_label,
            "reconstruction_error": score,
            "pred_label": pred_label,
            "detection_method": method,
            "cut_detected": cut_detected,
            "correct": true_label == pred_label
        })

    # Step 6: Save and analyze results
    results_df = pd.DataFrame(results)
    results_df.to_csv("sensor_classification_results.csv", index=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    accuracy = results_df['correct'].mean() * 100

    tp = len(results_df[(results_df['true_label'] == 'defective') & (results_df['pred_label'] == 'defective')])
    tn = len(results_df[(results_df['true_label'] == 'nondefective') & (results_df['pred_label'] == 'nondefective')])
    fp = len(results_df[(results_df['true_label'] == 'nondefective') & (results_df['pred_label'] == 'defective')])
    fn = len(results_df[(results_df['true_label'] == 'defective') & (results_df['pred_label'] == 'nondefective')])

    # Detection method breakdown
    cut_detections = len(results_df[results_df['detection_method'] == 'cut_detection'])
    ae_detections = len(results_df[results_df['detection_method'] == 'autoencoder'])

    print(f"\nOverall Accuracy: {accuracy:.1f}%")
    print(f"\nDetection Method Breakdown:")
    print(f"  Cut Detection: {cut_detections} sensors")
    print(f"  Autoencoder: {ae_detections} sensors")

    print(f"\nConfusion Matrix:")
    print(f"  True Positives (Defective detected): {tp}")
    print(f"  True Negatives (Non-defective correct): {tn}")
    print(f"  False Positives (False alarms): {fp}")
    print(f"  False Negatives (Missed defects): {fn}")

    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"\nPrecision: {precision:.2%}")
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"Recall (Detection rate): {recall:.2%}")

    print(f"\n✓ Results saved to sensor_classification_results.csv")

    # Step 7: Visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    visualize_results(results_df, NONDEF_DIR, DEF_DIR, num_samples=50)
    create_score_histogram(results_df, threshold)


if __name__ == "__main__":
    main()
