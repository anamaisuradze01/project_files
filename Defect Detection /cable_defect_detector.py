import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json


class CableDefectDetection:
    """
    Cable Defect Detection using Autoencoder and Pseudo Defect Generation
    
    This model performs two-stage detection:
    1. Extract individual cables from sensor images
    2. Classify each cable as defective or non-defective
    
    Detection methods:
    - Cut detection: Detects diagonal/vertical cuts using Hough Line Transform
    - Autoencoder: Detects large spots, burns, and other anomalies
    """
    
    def __init__(self, results_dir, model_path=None, **kwargs):
        """
        Initialize the cable defect detection model.
        
        Args:
            results_dir: Directory to save results and visualizations
            model_path: Path to the trained autoencoder weights (.keras file)
            **kwargs: Additional configuration options
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Configuration
        self.patch_size = kwargs.get('patch_size', 64)
        self.stride = kwargs.get('stride', 32)
        self.min_cut_angle = kwargs.get('min_cut_angle', 30)
        self.threshold = kwargs.get('threshold', 0.0035)
        self.min_sensor_width_ratio = kwargs.get('min_sensor_width_ratio', 0.3)
        
        # Load autoencoder model
        if model_path is None:
            model_path = kwargs.get('autoencoder_path', 'sensor_autoencoder.keras')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        self.model = tf.keras.models.load_model(model_path)
        print(f"✓ Loaded autoencoder model from {model_path}")
    
    # ====================================
    # CABLE EXTRACTION
    # ====================================
    
    def extract_cables_from_image(self, image_path):
        """
        Extract individual cables from sensor image using contour detection.
        
        Returns:
            cables: List of cable images (numpy arrays)
            cable_info: List of metadata dictionaries for each cable
        """
        img = cv2.imread(image_path)
        if img is None:
            return [], []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and extract cables
        cables = []
        cable_info = []
        
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter: must be large enough and relatively horizontal
            if area > 1000 and w > width * self.min_sensor_width_ratio:
                # Extract cable region with small padding
                pad = 5
                y1 = max(0, y - pad)
                y2 = min(height, y + h + pad)
                x1 = max(0, x - pad)
                x2 = min(width, x + w + pad)
                
                cable_img = img[y1:y2, x1:x2]
                cables.append(cable_img)
                cable_info.append({
                    'cable_idx': idx,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'extracted_region': (x1, y1, x2, y2)
                })
        
        # Sort cables by vertical position (top to bottom)
        sorted_indices = sorted(range(len(cables)), key=lambda i: cable_info[i]['bbox'][1])
        cables = [cables[i] for i in sorted_indices]
        cable_info = [cable_info[i] for i in sorted_indices]
        
        # Renumber cables from top to bottom
        for i, info in enumerate(cable_info):
            info['cable_idx'] = i
        
        return cables, cable_info
    
    # ====================================
    # CUT DETECTION
    # ====================================
    
    def preprocess_for_cut_detection(self, img):
        """Preprocessing for detecting cuts/lines using edge detection."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        return edges
    
    def detect_cuts_hough(self, img):
        """
        Detect diagonal or vertical cuts using Hough Line Transform.
        
        Returns:
            bool: True if diagonal/vertical cuts detected
            list: List of detected line angles
        """
        edges = self.preprocess_for_cut_detection(img)
        
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
            
            # Check if line is diagonal or vertical (>= min_angle degrees)
            if angle >= self.min_cut_angle:
                cut_detected = True
        
        return cut_detected, detected_angles
    
    # ====================================
    # AUTOENCODER PREPROCESSING
    # ====================================
    
    def preprocess_sensor(self, img):
        """Minimal preprocessing to preserve defects for autoencoder."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def resize_sensor(self, img, target_height=128):
        """Resize sensor to standard height while maintaining aspect ratio."""
        h, w = img.shape[:2]
        if h == 0:
            return img
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
        return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    def extract_patches(self, img):
        """Extract overlapping patches from cable image."""
        patches = []
        h, w = img.shape[:2]
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = img[y:y + self.patch_size, x:x + self.patch_size]
                if patch.std() > 2:
                    patches.append(patch)
        
        return patches
    
    def patches_to_numpy(self, patches):
        """Convert patches to numpy array for model input."""
        if len(patches) == 0:
            return np.array([]).reshape(0, self.patch_size, self.patch_size, 1)
        arr = np.stack(patches).astype("float32") / 255.0
        arr = arr[..., np.newaxis]
        return arr
    
    # ====================================
    # CABLE CLASSIFICATION
    # ====================================
    
    def classify_cable(self, cable_img):
        """
        Classify a single cable as defective or non-defective.
        
        Two-stage detection:
        1. Check for diagonal/vertical cuts
        2. Use autoencoder for other defects (spots, burns)
        
        Returns:
            dict: Classification results with score and method
        """
        # Stage 1: Cut detection
        cut_detected, angles = self.detect_cuts_hough(cable_img)
        
        if cut_detected:
            return {
                'classification': 'Defective',
                'score': 999.0,
                'method': 'cut_detection',
                'cut_detected': True,
                'angles': angles
            }
        
        # Stage 2: Autoencoder for other defects
        preprocessed = self.preprocess_sensor(cable_img)
        resized = self.resize_sensor(preprocessed)
        patches = self.extract_patches(resized)
        
        if len(patches) == 0:
            return {
                'classification': 'Good',
                'score': 0.0,
                'method': 'autoencoder',
                'cut_detected': False,
                'angles': []
            }
        
        X = self.patches_to_numpy(patches)
        recon = self.model.predict(X, batch_size=128, verbose=0)
        errors = np.mean(np.square(recon - X), axis=(1, 2, 3))
        
        # Use maximum error (focuses on worst patch where defect is)
        max_error = float(np.max(errors))
        
        classification = 'Defective' if max_error > self.threshold else 'Good'
        
        return {
            'classification': classification,
            'score': max_error,
            'method': 'autoencoder',
            'cut_detected': False,
            'angles': []
        }
    
    # ====================================
    # VISUALIZATION
    # ====================================
    
    def create_visualization(self, image_path, cables, cable_info, classifications, run_id):
        """Create visualization showing extracted cables and classifications."""
        img = cv2.imread(image_path)
        vis_img = img.copy()
        
        # Draw bounding boxes and labels on original image
        for info, result in zip(cable_info, classifications):
            x, y, w, h = info['bbox']
            
            # Color code based on classification
            if result['classification'] == 'Defective':
                color = (0, 0, 255)  # Red
                label_color = (0, 0, 255)
            else:
                color = (0, 255, 0)  # Green
                label_color = (0, 255, 0)
            
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
            
            # Add label with classification
            label = f"C{info['cable_idx']}: {result['classification']}"
            if result['method'] == 'cut_detection':
                label += " (CUT)"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (x, y - text_h - 8), (x + text_w + 5, y), label_color, -1)
            cv2.putText(vis_img, label, (x + 2, y - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add summary text
        defective_count = sum(1 for r in classifications if r['classification'] == 'Defective')
        good_count = len(classifications) - defective_count
        
        summary = f"Total Cables: {len(classifications)} | Good: {good_count} | Defective: {defective_count}"
        cv2.putText(vis_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        
        # Save visualization
        vis_filename = f"{run_id}_visualization.jpg"
        vis_path = os.path.join(self.results_dir, vis_filename)
        cv2.imwrite(vis_path, vis_img)
        
        return vis_filename
    
    # ====================================
    # MAIN PREDICTION METHOD
    # ====================================
    
    def predict(self, image_path, run_id):
        """
        Main prediction method that extracts cables and classifies them.
        
        Args:
            image_path: Path to the uploaded sensor image
            run_id: Unique string prefix for output files
        
        Returns:
            dict: Results containing classification, attributes, and visualization
        """
        # Read image info
        img = cv2.imread(image_path)
        if img is None:
            return {
                'output': 'Error: Could not read image',
                'attributes': {},
                'visualization_filename': None,
                'image_info': {
                    'filename': os.path.basename(image_path),
                    'width': 0,
                    'height': 0
                }
            }
        
        h, w = img.shape[:2]
        
        # Extract cables from image
        cables, cable_info = self.extract_cables_from_image(image_path)
        
        if len(cables) == 0:
            return {
                'output': 'No cables detected in image',
                'attributes': {
                    'Total Cables': 0,
                    'Defective': 0,
                    'Good': 0
                },
                'visualization_filename': None,
                'image_info': {
                    'filename': os.path.basename(image_path),
                    'width': w,
                    'height': h
                }
            }
        
        # Classify each cable
        classifications = []
        for cable in cables:
            result = self.classify_cable(cable)
            classifications.append(result)
        
        # Count results
        defective_count = sum(1 for r in classifications if r['classification'] == 'Defective')
        good_count = len(classifications) - defective_count
        
        # Count detection methods
        cut_detections = sum(1 for r in classifications if r['method'] == 'cut_detection')
        ae_detections = sum(1 for r in classifications if r['method'] == 'autoencoder')
        
        # Calculate average reconstruction error (excluding cut detections)
        ae_scores = [r['score'] for r in classifications if r['method'] == 'autoencoder']
        avg_score = float(np.mean(ae_scores)) if ae_scores else 0.0
        max_score = float(np.max(ae_scores)) if ae_scores else 0.0
        
        # Create visualization
        vis_filename = self.create_visualization(image_path, cables, cable_info, 
                                                 classifications, run_id)
        
        # Determine overall output
        if defective_count > 0:
            output = f"{defective_count} Defective cable(s) detected"
        else:
            output = "All cables are Good"
        
        # Save detailed results to JSON
        results_data = {
            'total_cables': len(cables),
            'defective_count': defective_count,
            'good_count': good_count,
            'cable_classifications': [
                {
                    'cable_idx': info['cable_idx'],
                    'classification': result['classification'],
                    'score': result['score'],
                    'method': result['method'],
                    'cut_detected': result['cut_detected']
                }
                for info, result in zip(cable_info, classifications)
            ]
        }
        
        results_json = f"{run_id}_results.json"
        with open(os.path.join(self.results_dir, results_json), 'w') as f:
            json.dump(results_data, f, indent=2)
        
        return {
            'output': output,
            'attributes': {
                'Total Cables': len(cables),
                'Defective': defective_count,
                'Good': good_count,
                'Cut Detections': cut_detections,
                'Autoencoder Detections': ae_detections,
                'Avg Reconstruction Error': round(avg_score, 6),
                'Max Reconstruction Error': round(max_score, 6),
                'Threshold': self.threshold
            },
            'visualization_filename': vis_filename,
            'image_info': {
                'filename': os.path.basename(image_path),
                'width': w,
                'height': h
            }
        }
