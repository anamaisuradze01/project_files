import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class CableDefectDetector:
    def __init__(self, results_dir, model_path=None, **kwargs):
        """
        Initialize the Cable Defect Detector
        
        Args:
            results_dir: Directory to save visualization results
            model_path: Path to the trained autoencoder model (.keras file)
            **kwargs: Additional configuration (threshold, patch_size, stride, min_angle)
        """
        self.results_dir = results_dir
        
        # Configuration
        self.patch_size = kwargs.get('patch_size', 64)
        self.stride = kwargs.get('stride', 32)
        self.threshold = kwargs.get('threshold', 0.0035)
        self.min_cut_angle = kwargs.get('min_angle', 30)
        
        # Load the autoencoder model
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"✓ Loaded model from {model_path}")
        else:
            # Build a new lightweight model if no pretrained model provided
            print("⚠ No pretrained model found. Building new model...")
            self.model = self._build_lightweight_autoencoder()
            print("⚠ Warning: Model is not trained! Please provide a trained model_path.")
    
    def _build_lightweight_autoencoder(self):
        """Build the lightweight autoencoder architecture"""
        input_shape = (self.patch_size, self.patch_size, 1)
        inp = layers.Input(shape=input_shape)

        # Encoder
        x = layers.Conv2D(16, 3, activation='relu', padding='same', strides=2)(inp)
        x = layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same', strides=2)(x)

        x = layers.Flatten()(x)
        latent = layers.Dense(128, activation='relu')(x)

        # Decoder
        x = layers.Dense(8 * 8 * 64, activation='relu')(latent)
        x = layers.Reshape((8, 8, 64))(x)
        x = layers.Conv2DTranspose(32, 3, activation='relu', padding='same', strides=2)(x)
        x = layers.Conv2DTranspose(16, 3, activation='relu', padding='same', strides=2)(x)
        x = layers.Conv2DTranspose(8, 3, activation='relu', padding='same', strides=2)(x)

        out = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

        model = models.Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

        return model
    
    def _preprocess_for_cut_detection(self, img):
        """Preprocessing for cut/line detection"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        return edges
    
    def _detect_cuts_hough(self, img):
        """Detect diagonal or vertical cuts using Hough Line Transform"""
        edges = self._preprocess_for_cut_detection(img)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=50,
            maxLineGap=10
        )

        if lines is None:
            return False, []

        detected_angles = []
        cut_detected = False

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 - x1 == 0:
                angle = 90
            else:
                angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))

            detected_angles.append(angle)

            if angle >= self.min_cut_angle:
                cut_detected = True

        return cut_detected, detected_angles
    
    def _preprocess_sensor(self, img):
        """Minimal preprocessing for autoencoder"""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def _resize_sensor(self, img, target_height=128):
        """Resize sensor maintaining aspect ratio"""
        h, w = img.shape[:2]
        if h == 0:
            return img
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
        return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    def _extract_patches(self, img):
        """Extract overlapping patches from sensor image"""
        patches = []
        h, w = img.shape[:2]

        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = img[y:y + self.patch_size, x:x + self.patch_size]
                if patch.std() > 2:
                    patches.append(patch)

        return patches
    
    def _patches_to_numpy(self, patches):
        """Convert patches to numpy array"""
        if len(patches) == 0:
            return np.array([]).reshape(0, self.patch_size, self.patch_size, 1)
        arr = np.stack(patches).astype("float32") / 255.0
        arr = arr[..., np.newaxis]
        return arr
    
    def _create_visualization(self, img, cut_detected, score, detection_method, output_path):
        """Create visualization with detection results overlaid"""
        vis_img = img.copy()
        h, w = vis_img.shape[:2]
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        if cut_detected:
            text = "DEFECTIVE: Cut Detected"
            color = (0, 0, 255)  # Red
            
            # Draw detected lines
            edges = self._preprocess_for_cut_detection(img)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if x2 - x1 == 0:
                        angle = 90
                    else:
                        angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))
                    
                    if angle >= self.min_cut_angle:
                        cv2.line(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        elif score > self.threshold:
            text = "DEFECTIVE: Anomaly Detected"
            color = (0, 165, 255)  # Orange
        else:
            text = "NON-DEFECTIVE"
            color = (0, 255, 0)  # Green
        
        # Add text background
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(vis_img, (10, 10), (20 + text_w, 30 + text_h), (0, 0, 0), -1)
        cv2.putText(vis_img, text, (15, 40), font, font_scale, color, thickness)
        
        # Add score
        score_text = f"Score: {score:.4f}" if score < 900 else "Score: CUT"
        cv2.putText(vis_img, score_text, (15, 80), font, 0.7, (255, 255, 255), 2)
        
        # Save visualization
        cv2.imwrite(output_path, vis_img)
        
        return vis_img

    def predict(self, image_path, run_id):
        """
        Main prediction method
        
        Args:
            image_path: Path to the uploaded image
            run_id: Unique string prefix for output files
        
        Returns:
            dict: Results dictionary with output, attributes, visualization, and image info
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return {
                'output': 'Error: Could not load image',
                'attributes': {},
                'visualization_filename': None,
                'image_info': {
                    'filename': os.path.basename(image_path),
                    'width': 0,
                    'height': 0
                }
            }
        
        h, w = img.shape[:2]
        
        # STAGE 1: Cut Detection
        cut_detected, angles = self._detect_cuts_hough(img)
        
        if cut_detected:
            # Cut detected - classify as defective immediately
            score = 999.0
            detection_method = "cut_detection"
            output = "Defective"
            max_angle = max(angles) if angles else 0
            
            attributes = {
                'Detection Method': 'Cut Detection',
                'Reconstruction Score': 999.0,
                'Threshold': self.threshold,
                'Cut Detected': 'Yes',
                'Max Cut Angle': f"{max_angle:.1f}°",
                'Status': 'DEFECTIVE - Cut/Crack Found'
            }
        else:
            # STAGE 2: Autoencoder for other defects
            preprocessed = self._preprocess_sensor(img)
            resized = self._resize_sensor(preprocessed)
            patches = self._extract_patches(resized)
            
            if len(patches) == 0:
                score = 0.0
                detection_method = "autoencoder"
                output = "Non-Defective"
            else:
                X = self._patches_to_numpy(patches)
                recon = self.model.predict(X, batch_size=128, verbose=0)
                errors = np.mean(np.square(recon - X), axis=(1, 2, 3))
                score = float(np.max(errors))
                detection_method = "autoencoder"
                
                # Classify based on threshold
                output = "Defective" if score > self.threshold else "Non-Defective"
            
            attributes = {
                'Detection Method': 'Autoencoder',
                'Reconstruction Score': round(score, 6),
                'Threshold': self.threshold,
                'Cut Detected': 'No',
                'Patches Analyzed': len(patches),
                'Status': 'DEFECTIVE - Anomaly' if score > self.threshold else 'GOOD'
            }
        
        # Create visualization
        vis_filename = f"{run_id}_vis.jpg"
        vis_path = os.path.join(self.results_dir, vis_filename)
        self._create_visualization(img, cut_detected, score, detection_method, vis_path)
        
        return {
            'output': output,
            'attributes': attributes,
            'visualization_filename': vis_filename,
            'image_info': {
                'filename': os.path.basename(image_path),
                'width': w,
                'height': h
            }
        }
