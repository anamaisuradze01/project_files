import os
import cv2
import numpy as np
from scipy.stats import norm


class CableMeasurement:
    def __init__(self, results_dir, **kwargs):
        """
        Initialize the Cable Measurement model.
        
        Args:
            results_dir: Directory where output files will be saved
            **kwargs: Additional configuration parameters (not used in this model)
        """
        self.results_dir = results_dir
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _detect_sensor_contours(self, img):
        """Detect cable contours in the image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Apply binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply morphological closing to fill small gaps and remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Filter contours to find cables (large horizontal regions)
        cable_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # Filter: must be large enough and relatively horizontal
            if area > 1000 and w > width * 0.3:  # At least 30% of image width
                cable_contours.append({
                    'contour': contour,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'area': area,
                    'center_y': y + h // 2
                })

        # Sort cables by vertical position (top to bottom)
        cable_contours.sort(key=lambda s: s['y'])
        
        return cable_contours

    def _extract_contour_boundaries(self, contour, x_min, x_max):
        """Extract upper and lower boundaries from a contour."""
        upper_boundary = {}
        lower_boundary = {}

        # Extract all points from contour
        points = contour.reshape(-1, 2)

        for point in points:
            x, y = point
            if x_min <= x <= x_max:
                # Update upper boundary (minimum y for each x) - top edge
                if x not in upper_boundary or y < upper_boundary[x]:
                    upper_boundary[x] = y
                # Update lower boundary (maximum y for each x) - bottom edge
                if x not in lower_boundary or y > lower_boundary[x]:
                    lower_boundary[x] = y

        return upper_boundary, lower_boundary

    def _interpolate_boundary(self, boundary_dict, x_min, x_max):
        """Interpolate missing x-coordinates in boundary using linear interpolation."""
        x_coords = sorted(boundary_dict.keys())
        if len(x_coords) < 2:
            return boundary_dict

        # Fill in missing x values using interpolation
        complete_boundary = {}
        for x in range(x_min, x_max + 1):
            if x in boundary_dict:
                complete_boundary[x] = boundary_dict[x]
            else:
                # Find nearest points on left and right
                left_x = max([xc for xc in x_coords if xc < x], default=None)
                right_x = min([xc for xc in x_coords if xc > x], default=None)

                if left_x is not None and right_x is not None:
                    # Linear interpolation
                    t = (x - left_x) / (right_x - left_x)
                    y = boundary_dict[left_x] * (1 - t) + boundary_dict[right_x] * t
                    complete_boundary[x] = int(y)
                elif left_x is not None:
                    complete_boundary[x] = boundary_dict[left_x]
                elif right_x is not None:
                    complete_boundary[x] = boundary_dict[right_x]

        return complete_boundary

    def _analyze_cable_thickness(self, cable):
        """Measure the vertical distance between upper and lower contour at each x position."""
        contour = cable['contour']
        x, y, w, h = cable['x'], cable['y'], cable['w'], cable['h']

        # Extract upper and lower boundaries from the cable's contour
        upper_boundary, lower_boundary = self._extract_contour_boundaries(contour, x, x + w - 1)

        # Interpolate to fill missing x-coordinates
        upper_boundary = self._interpolate_boundary(upper_boundary, x, x + w - 1)
        lower_boundary = self._interpolate_boundary(lower_boundary, x, x + w - 1)

        # For each x-position, calculate vertical distance between upper and lower contour
        # Skip first and last x positions to avoid vertical borders
        skip_pixels = 5
        thicknesses = []
        thickness_data = []

        for col_x in range(x + skip_pixels, x + w - skip_pixels):
            if col_x in upper_boundary and col_x in lower_boundary:
                upper_y = upper_boundary[col_x]
                lower_y = lower_boundary[col_x]
                thickness = abs(lower_y - upper_y)

                thicknesses.append(thickness)
                thickness_data.append({
                    'col': col_x,
                    'thickness': thickness,
                    'top': upper_y,
                    'bottom': lower_y
                })

        if len(thicknesses) == 0:
            return None

        # Find minimum, maximum, and average thickness
        min_thickness = min(thicknesses)
        max_thickness = max(thicknesses)
        avg_thickness = np.mean(thicknesses)
        std_thickness = np.std(thicknesses)

        # Find where minimum occurs
        min_idx = thicknesses.index(min_thickness)
        min_data = thickness_data[min_idx]

        # Find where maximum occurs
        max_idx = thicknesses.index(max_thickness)
        max_data = thickness_data[max_idx]

        # Find where thickness is closest to average
        closest_to_avg_idx = min(range(len(thicknesses)),
                                 key=lambda i: abs(thicknesses[i] - avg_thickness))
        avg_data = thickness_data[closest_to_avg_idx]

        return {
            'min_thickness': min_thickness,
            'max_thickness': max_thickness,
            'avg_thickness': avg_thickness,
            'std_thickness': std_thickness,
            'all_thicknesses': thicknesses,
            'min_col': min_data['col'],
            'min_top': min_data['top'],
            'min_bottom': min_data['bottom'],
            'max_col': max_data['col'],
            'max_top': max_data['top'],
            'max_bottom': max_data['bottom'],
            'avg_col': avg_data['col'],
            'avg_top': avg_data['top'],
            'avg_bottom': avg_data['bottom']
        }

    def _draw_measurements(self, img, cables, thickness_data):
        """Draw the contours and measurement arrows on the image."""
        output = img.copy()

        if len(cables) == 0:
            return output

        # Colors
        min_color = (255, 0, 255)  # Magenta for minimum
        max_color = (0, 255, 255)  # Cyan for maximum
        avg_color = (0, 0, 255)  # Blue for average
        contour_color = (0, 255, 0)  # Green for contours
        distance_color = (0, 100, 255)  # Orange for distances between cables
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw each cable
        for i, (cable, thickness) in enumerate(zip(cables, thickness_data)):
            if thickness is None:
                continue

            # Draw the contour
            cv2.drawContours(output, [cable['contour']], -1, contour_color, 2)

            # Draw MIN thickness arrow (magenta)
            cv2.arrowedLine(output,
                            (thickness['min_col'], thickness['min_top']),
                            (thickness['min_col'], thickness['min_bottom']),
                            min_color, 3, tipLength=0.05)
            cv2.arrowedLine(output,
                            (thickness['min_col'], thickness['min_bottom']),
                            (thickness['min_col'], thickness['min_top']),
                            min_color, 3, tipLength=0.05)

            text = f"MIN: {thickness['min_thickness']}px"
            text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
            text_x = thickness['min_col'] + 10
            text_y = (thickness['min_top'] + thickness['min_bottom']) // 2

            cv2.rectangle(output,
                          (text_x - 3, text_y - text_size[1] - 3),
                          (text_x + text_size[0] + 3, text_y + 3),
                          bg_color, -1)
            cv2.putText(output, text, (text_x, text_y),
                        font, 0.6, min_color, 2)

            # Draw MAX thickness arrow (cyan)
            cv2.arrowedLine(output,
                            (thickness['max_col'], thickness['max_top']),
                            (thickness['max_col'], thickness['max_bottom']),
                            max_color, 3, tipLength=0.05)
            cv2.arrowedLine(output,
                            (thickness['max_col'], thickness['max_bottom']),
                            (thickness['max_col'], thickness['max_top']),
                            max_color, 3, tipLength=0.05)

            text = f"MAX: {thickness['max_thickness']}px"
            text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
            text_x = thickness['max_col'] + 10
            text_y = (thickness['max_top'] + thickness['max_bottom']) // 2

            cv2.rectangle(output,
                          (text_x - 3, text_y - text_size[1] - 3),
                          (text_x + text_size[0] + 3, text_y + 3),
                          bg_color, -1)
            cv2.putText(output, text, (text_x, text_y),
                        font, 0.6, max_color, 2)

            # Draw AVG thickness arrow (blue)
            cv2.arrowedLine(output,
                            (thickness['avg_col'], thickness['avg_top']),
                            (thickness['avg_col'], thickness['avg_bottom']),
                            avg_color, 4, tipLength=0.05)
            cv2.arrowedLine(output,
                            (thickness['avg_col'], thickness['avg_bottom']),
                            (thickness['avg_col'], thickness['avg_top']),
                            avg_color, 4, tipLength=0.05)

            text = f"AVG: {thickness['avg_thickness']:.1f}px (σ={thickness['std_thickness']:.1f})"
            text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
            text_x = thickness['avg_col'] + 15
            text_y = (thickness['avg_top'] + thickness['avg_bottom']) // 2

            cv2.rectangle(output,
                          (text_x - 3, text_y - text_size[1] - 3),
                          (text_x + text_size[0] + 3, text_y + 3),
                          bg_color, -1)
            cv2.putText(output, text, (text_x, text_y),
                        font, 0.7, avg_color, 2)

            # Draw cable label
            label = f"Cable {i + 1}"
            cv2.putText(output, label, (cable['x'] + 10, cable['y'] + 20),
                        font, 0.8, text_color, 2)

        # Draw distances between cables
        for i in range(len(cables) - 1):
            current = cables[i]
            next_cable = cables[i + 1]

            # Distance from bottom of current to top of next
            distance = next_cable['y'] - (current['y'] + current['h'])

            x_pos = 80
            start_y = current['y'] + current['h']
            end_y = next_cable['y']
            mid_y = (start_y + end_y) // 2

            cv2.arrowedLine(output,
                            (x_pos, start_y),
                            (x_pos, end_y),
                            distance_color, 2, tipLength=0.03)
            cv2.arrowedLine(output,
                            (x_pos, end_y),
                            (x_pos, start_y),
                            distance_color, 2, tipLength=0.03)

            text = f"D: {distance}px"
            text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
            text_x = x_pos - text_size[0] - 15
            text_y = mid_y + text_size[1] // 2

            cv2.rectangle(output,
                          (text_x - 2, text_y - text_size[1] - 2),
                          (text_x + text_size[0] + 2, text_y + 2),
                          bg_color, -1)
            cv2.putText(output, text, (text_x, text_y),
                        font, 0.5, text_color, 1)

        return output

    def predict(self, image_path, run_id):
        """
        Analyze cable thickness measurements from an image.
        
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
        
        # Detect cables
        cables = self._detect_sensor_contours(img)
        
        # Analyze thickness for each cable
        thickness_data = []
        for cable in cables:
            thickness = self._analyze_cable_thickness(cable)
            thickness_data.append(thickness)
        
        # Calculate combined statistics for all cables
        all_combined_thicknesses = []
        for thickness in thickness_data:
            if thickness is not None:
                all_combined_thicknesses.extend(thickness['all_thicknesses'])
        
        # Determine output status
        num_cables = len([t for t in thickness_data if t is not None])
        
        if num_cables == 0:
            output = "No cables detected"
            combined_mean = 0
            combined_std = 0
            combined_min = 0
            combined_max = 0
            combined_cv = 0
        else:
            output = f"{num_cables} cable{'s' if num_cables > 1 else ''} detected - Measurements completed"
            combined_mean = np.mean(all_combined_thicknesses)
            combined_std = np.std(all_combined_thicknesses)
            combined_min = np.min(all_combined_thicknesses)
            combined_max = np.max(all_combined_thicknesses)
            combined_cv = (combined_std / combined_mean) * 100 if combined_mean > 0 else 0
        
        # Create visualization
        vis_img = self._draw_measurements(img, cables, thickness_data)
        vis_filename = f"{run_id}_vis.jpg"
        cv2.imwrite(os.path.join(self.results_dir, vis_filename), vis_img)
        
        # Build attributes dictionary
        attributes = {
            'Cables Detected': num_cables,
            'Combined Mean Thickness (px)': round(combined_mean, 2),
            'Combined Std Dev (px)': round(combined_std, 2),
            'Combined Min Thickness (px)': int(combined_min) if combined_min > 0 else 0,
            'Combined Max Thickness (px)': int(combined_max) if combined_max > 0 else 0,
            'Combined CV (%)': round(combined_cv, 2),
            'Total Data Points': len(all_combined_thicknesses)
        }
        
        # Add individual cable measurements
        for i, thickness in enumerate(thickness_data):
            if thickness is not None:
                attributes[f'Cable {i+1} Avg (px)'] = round(thickness['avg_thickness'], 2)
                attributes[f'Cable {i+1} Std Dev (px)'] = round(thickness['std_thickness'], 2)
                attributes[f'Cable {i+1} Min (px)'] = thickness['min_thickness']
                attributes[f'Cable {i+1} Max (px)'] = thickness['max_thickness']
        
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
