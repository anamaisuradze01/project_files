import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import random


class CableDefectDetection:
    def __init__(self, results_dir, **kwargs):
        """
        Initialize the Cable Defect Detection model.
        
        Args:
            results_dir: Directory where output files will be saved
            **kwargs: Additional configuration parameters
                - model_path: Path to the trained autoencoder model (.keras file)
                - threshold: Detection threshold (default: 0.0035)
                - patch_size: Patch size for autoencoder (default: 64)
                - stride: Stride for patch extraction (default: 32)
                - min_cut_angle: Minimum angle for cut detection in degrees (default: 30)
                - min_sensor_width_ratio: Minimum width ratio for cable detection (default: 0.3)
        """
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Model parameters
        self.threshold = kwargs.get('threshold', 0.0035)
        self.patch_size = kwargs.get('patch_size', 64)
        self.stride = kwargs.get('stride', 32)
        self.min_cut_angle = kwargs.get('min_cut_angle', 30)
        self.min_sensor_width_ratio = kwargs.get('min_sensor_width_ratio', 0.3)
        
        # Load the autoencoder model
        model_path = kwargs.get('model_path', 'sensor_autoencoder.keras')
        if os.path.exists(model_path):
            self.model = models.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Model will be unavailable.")
            self.model = None
    
    def _extract_sensors_from_image(self, img):
        """Extract individual cables from an image using contour detection."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and extract sensors
        sensors = []
        sensor_info = []

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # Filter: must be large enough and relatively horizontal
            if area > 1000 and w > width * self.min_sensor_width_ratio:
                # Extract sensor region with small padding
                pad = 5
                y1 = max(0, y - pad)
                y2 = min(height, y + h + pad)
                x1 = max(0, x - pad)
                x2 = min(width, x + w + pad)

                sensor_img = img[y1:y2, x1:x2]
                sensors.append(sensor_img)
                sensor_info.append({
                    'sensor_idx': idx,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'extracted_region': (x1, y1, x2, y2)
                })

        # Sort sensors by vertical position (top to bottom)
        sorted_indices = sorted(range(len(sensors)), key=lambda i: sensor_info[i]['bbox'][1])
        sensors = [sensors[i] for i in sorted_indices]
        sensor_info = [sensor_info[i] for i in sorted_indices]

        # Renumber sensors from top to bottom
        for i, info in enumerate(sensor_info):
            info['sensor_idx'] = i

        return sensors, sensor_info
    
    def _preprocess_for_cut_detection(self, img):
        """Special preprocessing for detecting cuts/lines."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        return edges
    
    def _detect_cuts_hough(self, img):
        """Detect diagonal or vertical cuts using Hough Line Transform."""
        edges = self._preprocess_for_cut_detection(img)

        # Detect lines using Hough Transform
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

            # Calculate angle from horizontal
            if x2 - x1 == 0:
                angle = 90
            else:
                angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))

            detected_angles.append(angle)

            # Check if line is diagonal or vertical
            if angle >= self.min_cut_angle:
                cut_detected = True

        return cut_detected, detected_angles
    
    def _preprocess_sensor(self, img):
        """Minimal preprocessing - preserve features."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def _resize_sensor(self, img, target_height=128):
        """Resize sensor to standard height while maintaining aspect ratio."""
        h, w = img.shape[:2]
        if h == 0:
            return img
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
        return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    def _extract_patches(self, img):
        """Extract overlapping patches from sensor image."""
        patches = []
        h, w = img.shape[:2]

        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = img[y:y + self.patch_size, x:x + self.patch_size]
                if patch.std() > 2:
                    patches.append(patch)

        return patches
    
    def _patches_to_numpy(self, patches):
        """Convert patches to numpy array."""
        if len(patches) == 0:
            return np.array([]).reshape(0, 64, 64, 1)
        arr = np.stack(patches).astype("float32") / 255.0
        arr = arr[..., np.newaxis]
        return arr
    
    def _compute_sensor_score(self, sensor_img):
        """
        Two-stage classification:
        1. Check for diagonal/vertical cuts
        2. If no cuts, use autoencoder for other defects
        
        Returns:
            tuple: (score, detection_method, cut_detected)
        """
        # STAGE 1: Cut Detection
        cut_detected, angles = self._detect_cuts_hough(sensor_img)

        if cut_detected:
            return 999.0, "cut_detection", True

        # STAGE 2: Autoencoder for other defects
        if self.model is None:
            return 0.0, "error", False
        
        preprocessed = self._preprocess_sensor(sensor_img)
        resized = self._resize_sensor(preprocessed)
        patches = self._extract_patches(resized)

        if len(patches) == 0:
            return 0.0, "autoencoder", False

        X = self._patches_to_numpy(patches)
        recon = self.model.predict(X, batch_size=128, verbose=0)
        errors = np.mean(np.square(recon - X), axis=(1, 2, 3))

        return np.max(errors), "autoencoder", False
    
    def _draw_cable_boxes(self, img, sensor_info, classifications):
        """Draw bounding boxes with classification results on the image."""
        vis_img = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for info, classification in zip(sensor_info, classifications):
            x, y, w, h = info['bbox']
            
            # Color based on classification
            if classification['is_defective']:
                color = (0, 0, 255)  # Red for defective
                label = f"Cable {info['sensor_idx']}: DEFECTIVE"
            else:
                color = (0, 255, 0)  # Green for non-defective
                label = f"Cable {info['sensor_idx']}: OK"
            
            # Draw box
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 3)
            
            # Draw label background
            label_size = cv2.getTextSize(label, font, 0.7, 2)[0]
            cv2.rectangle(vis_img, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(vis_img, label, (x, y - 5), font, 0.7, (255, 255, 255), 2)
            
            # Draw score
            score_text = f"Score: {classification['score']:.4f}" if classification['method'] == 'autoencoder' else "Cut detected"
            cv2.putText(vis_img, score_text, (x, y + h + 20), font, 0.5, color, 1)
        
        return vis_img
    
    def predict(self, image_path, run_id):
        """
        Analyze cable defects from an image.
        
        Args:
            image_path: path to the uploaded image
            run_id: unique string prefix for output files
                 
        Returns:
            dict with output, attributes, visualization_filename, and image_info
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        h, w = img.shape[:2]
        
        # Extract cables from image
        sensors, sensor_info = self._extract_sensors_from_image(img)
        
        if len(sensors) == 0:
            # No cables detected
            vis_filename = f"{run_id}_no_cables.jpg"
            cv2.imwrite(os.path.join(self.results_dir, vis_filename), img)
            
            return {
                'output': 'No cables detected',
                'attributes': {
                    'Cables Detected': 0,
                    'Defective Cables': 0,
                    'Non-Defective Cables': 0,
                },
                'visualization_filename': vis_filename,
                'image_info': {
                    'filename': os.path.basename(image_path),
                    'width': w,
                    'height': h
                }
            }
        
        # Classify each cable
        classifications = []
        for sensor_img in sensors:
            score, method, cut_detected = self._compute_sensor_score(sensor_img)
            
            # Determine if defective
            if method == "cut_detection":
                is_defective = True
            else:
                is_defective = score > self.threshold
            
            classifications.append({
                'score': score,
                'method': method,
                'cut_detected': cut_detected,
                'is_defective': is_defective
            })
        
        # Count defective and non-defective
        num_defective = sum(1 for c in classifications if c['is_defective'])
        num_nondefective = len(classifications) - num_defective
        
        # Create visualization
        vis_img = self._draw_cable_boxes(img, sensor_info, classifications)
        vis_filename = f"{run_id}_vis.jpg"
        cv2.imwrite(os.path.join(self.results_dir, vis_filename), vis_img)
        
        # Determine overall output message
        if num_defective == 0:
            output = f"{len(sensors)} cable{'s' if len(sensors) > 1 else ''} detected - All OK"
        elif num_defective == len(sensors):
            output = f"{len(sensors)} cable{'s' if len(sensors) > 1 else ''} detected - All DEFECTIVE"
        else:
            output = f"{len(sensors)} cables detected - {num_defective} DEFECTIVE, {num_nondefective} OK"
        
        # Build attributes dictionary
        attributes = {
            'Cables Detected': len(sensors),
            'Defective Cables': num_defective,
            'Non-Defective Cables': num_nondefective,
            'Detection Threshold': self.threshold,
            'Cut Angle Threshold (°)': self.min_cut_angle,
        }
        
        # Add individual cable results
        for i, (info, classification) in enumerate(zip(sensor_info, classifications)):
            cable_num = i + 1
            status = "DEFECTIVE" if classification['is_defective'] else "OK"
            method = "Cut" if classification['method'] == 'cut_detection' else "Autoencoder"
            
            attributes[f'Cable {cable_num} Status'] = status
            attributes[f'Cable {cable_num} Method'] = method
            
            if classification['method'] == 'autoencoder':
                attributes[f'Cable {cable_num} Score'] = round(classification['score'], 6)
        
        # Add detection method breakdown
        cut_detections = sum(1 for c in classifications if c['method'] == 'cut_detection')
        ae_detections = sum(1 for c in classifications if c['method'] == 'autoencoder')
        
        attributes['Detections by Cut Method'] = cut_detections
        attributes['Detections by Autoencoder'] = ae_detections
        
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
