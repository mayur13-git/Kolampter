"""
Kolam Design Analyzer
====================

This module analyzes Kolam designs to identify their underlying mathematical principles
including symmetry, fractals, topology, and geometric patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
import cv2
from typing import List, Tuple, Dict, Any
import json


class KolamAnalyzer:
    """Analyzes Kolam designs to extract mathematical principles."""
    
    def __init__(self):
        self.symmetry_types = ['radial', 'bilateral', 'translational', 'rotational']
        self.pattern_types = ['fractal', 'tessellation', 'spiral', 'grid', 'freeform']
    
    def analyze_symmetry(self, pattern: np.ndarray) -> Dict[str, Any]:
        """
        Analyze different types of symmetry in the Kolam pattern.
        
        Args:
            pattern: 2D numpy array representing the Kolam pattern
            
        Returns:
            Dictionary containing symmetry analysis results
        """
        results = {}
        
        # Radial symmetry analysis
        center = (pattern.shape[0] // 2, pattern.shape[1] // 2)
        radial_symmetry = self._check_radial_symmetry(pattern, center)
        results['radial_symmetry'] = radial_symmetry
        
        # Bilateral symmetry analysis
        bilateral_symmetry = self._check_bilateral_symmetry(pattern)
        results['bilateral_symmetry'] = bilateral_symmetry
        
        # Rotational symmetry analysis
        rotational_symmetry = self._check_rotational_symmetry(pattern)
        results['rotational_symmetry'] = rotational_symmetry
        
        return results
    
    def _check_radial_symmetry(self, pattern: np.ndarray, center: Tuple[int, int]) -> Dict[str, Any]:
        """Check for radial symmetry around a center point."""
        height, width = pattern.shape
        cy, cx = center
        
        # Create distance and angle maps
        y, x = np.ogrid[:height, :width]
        distances = np.sqrt((x - cx)**2 + (y - cy)**2)
        angles = np.arctan2(y - cy, x - cx)
        
        # Group pixels by distance and check angular consistency
        max_distance = int(np.max(distances))
        symmetry_scores = []
        
        for r in range(1, max_distance + 1):
            mask = (distances >= r - 0.5) & (distances < r + 0.5)
            if np.sum(mask) > 0:
                angular_values = pattern[mask]
                # Check if values are consistent across angles
                if len(np.unique(angular_values)) <= 2:  # Allow for some variation
                    symmetry_scores.append(1.0)
                else:
                    symmetry_scores.append(0.0)
        
        radial_score = np.mean(symmetry_scores) if symmetry_scores else 0.0
        
        return {
            'score': radial_score,
            'center': center,
            'max_radius': max_distance,
            'is_radial': radial_score > 0.7
        }
    
    def _check_bilateral_symmetry(self, pattern: np.ndarray) -> Dict[str, Any]:
        """Check for bilateral (mirror) symmetry."""
        height, width = pattern.shape
        
        # Check horizontal symmetry
        horizontal_symmetry = 0
        if height % 2 == 0:
            top_half = pattern[:height//2, :]
            bottom_half = np.flipud(pattern[height//2:, :])
            horizontal_symmetry = np.mean(top_half == bottom_half)
        
        # Check vertical symmetry
        vertical_symmetry = 0
        if width % 2 == 0:
            left_half = pattern[:, :width//2]
            right_half = np.fliplr(pattern[:, width//2:])
            vertical_symmetry = np.mean(left_half == right_half)
        
        # Check diagonal symmetries
        diagonal_symmetry = 0
        if height == width:
            diagonal_symmetry = np.mean(pattern == pattern.T)
        
        max_symmetry = max(horizontal_symmetry, vertical_symmetry, diagonal_symmetry)
        
        return {
            'horizontal': horizontal_symmetry,
            'vertical': vertical_symmetry,
            'diagonal': diagonal_symmetry,
            'max_score': max_symmetry,
            'is_bilateral': max_symmetry > 0.8
        }
    
    def _check_rotational_symmetry(self, pattern: np.ndarray) -> Dict[str, Any]:
        """Check for rotational symmetry."""
        height, width = pattern.shape
        center = (height // 2, width // 2)
        
        # Test different rotation angles
        angles = [90, 120, 180, 270, 360]
        symmetry_scores = {}
        
        for angle in angles:
            if angle == 360:
                continue
                
            # Rotate the pattern
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(pattern, rotation_matrix, (width, height))
            
            # Compare with original
            similarity = np.mean(pattern == rotated)
            symmetry_scores[f'{angle}_degrees'] = similarity
        
        # Find the best rotational symmetry
        best_angle = max(symmetry_scores, key=symmetry_scores.get)
        best_score = symmetry_scores[best_angle]
        
        return {
            'scores': symmetry_scores,
            'best_angle': best_angle,
            'best_score': best_score,
            'is_rotational': best_score > 0.8
        }
    
    def analyze_fractal_properties(self, pattern: np.ndarray) -> Dict[str, Any]:
        """
        Analyze fractal properties of the Kolam pattern.
        
        Args:
            pattern: 2D numpy array representing the Kolam pattern
            
        Returns:
            Dictionary containing fractal analysis results
        """
        # Box-counting dimension
        box_dimension = self._calculate_box_dimension(pattern)
        
        # Self-similarity analysis
        self_similarity = self._analyze_self_similarity(pattern)
        
        # Recursive structure detection
        recursive_structure = self._detect_recursive_structure(pattern)
        
        return {
            'box_dimension': box_dimension,
            'self_similarity': self_similarity,
            'recursive_structure': recursive_structure,
            'is_fractal': box_dimension > 1.5 and self_similarity > 0.6
        }
    
    def _calculate_box_dimension(self, pattern: np.ndarray) -> float:
        """Calculate the box-counting dimension of the pattern."""
        # Convert to binary
        binary_pattern = (pattern > 0).astype(int)
        
        # Different box sizes
        box_sizes = [2, 4, 8, 16, 32, 64]
        counts = []
        
        for size in box_sizes:
            if size >= min(pattern.shape):
                break
                
            # Count boxes that contain at least one pixel
            count = 0
            for i in range(0, pattern.shape[0], size):
                for j in range(0, pattern.shape[1], size):
                    box = binary_pattern[i:i+size, j:j+size]
                    if np.any(box):
                        count += 1
            counts.append(count)
        
        if len(counts) < 2:
            return 1.0
        
        # Calculate slope of log-log plot
        log_sizes = np.log(box_sizes[:len(counts)])
        log_counts = np.log(counts)
        
        if len(log_sizes) > 1:
            slope = np.polyfit(log_sizes, log_counts, 1)[0]
            return -slope
        else:
            return 1.0
    
    def _analyze_self_similarity(self, pattern: np.ndarray) -> float:
        """Analyze self-similarity in the pattern."""
        # Divide pattern into quadrants and compare
        height, width = pattern.shape
        h_mid, w_mid = height // 2, width // 2
        
        quadrants = [
            pattern[:h_mid, :w_mid],
            pattern[:h_mid, w_mid:],
            pattern[h_mid:, :w_mid],
            pattern[h_mid:, w_mid:]
        ]
        
        similarities = []
        for i in range(len(quadrants)):
            for j in range(i + 1, len(quadrants)):
                # Resize to same dimensions for comparison
                q1, q2 = quadrants[i], quadrants[j]
                min_h = min(q1.shape[0], q2.shape[0])
                min_w = min(q1.shape[1], q2.shape[1])
                
                q1_resized = q1[:min_h, :min_w]
                q2_resized = q2[:min_h, :min_w]
                
                similarity = np.mean(q1_resized == q2_resized)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _detect_recursive_structure(self, pattern: np.ndarray) -> Dict[str, Any]:
        """Detect recursive or repeating structures in the pattern."""
        # Look for repeating motifs
        height, width = pattern.shape
        
        # Check for grid-like patterns
        grid_periodicity = self._find_grid_periodicity(pattern)
        
        # Check for spiral patterns
        spiral_detection = self._detect_spiral_pattern(pattern)
        
        return {
            'grid_periodicity': grid_periodicity,
            'spiral_detection': spiral_detection,
            'has_recursive_structure': grid_periodicity['has_grid'] or spiral_detection['has_spiral']
        }
    
    def _find_grid_periodicity(self, pattern: np.ndarray) -> Dict[str, Any]:
        """Find periodic grid structures in the pattern."""
        # Use autocorrelation to find repeating patterns
        autocorr = np.correlate(pattern.flatten(), pattern.flatten(), mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Find peaks in autocorrelation
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(autocorr[1:], height=np.max(autocorr) * 0.3)
        
        if len(peaks) > 0:
            period = peaks[0] + 1
            return {
                'has_grid': True,
                'period': period,
                'confidence': autocorr[period] / np.max(autocorr)
            }
        else:
            return {'has_grid': False, 'period': 0, 'confidence': 0.0}
    
    def _detect_spiral_pattern(self, pattern: np.ndarray) -> Dict[str, Any]:
        """Detect spiral patterns in the Kolam."""
        # Find the center
        center = (pattern.shape[0] // 2, pattern.shape[1] // 2)
        
        # Create polar coordinates
        y, x = np.ogrid[:pattern.shape[0], :pattern.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        theta = np.arctan2(y - center[0], x - center[1])
        
        # Check for spiral-like patterns
        # This is a simplified detection - more sophisticated methods could be used
        spiral_score = 0.0
        
        # Check if pattern follows a spiral-like distribution
        if np.max(r) > 0:
            # Normalize radius
            r_norm = r / np.max(r)
            
            # Check for increasing complexity with radius
            radial_complexity = []
            for radius in np.linspace(0.1, 1.0, 10):
                mask = (r_norm >= radius - 0.1) & (r_norm < radius + 0.1)
                if np.sum(mask) > 0:
                    complexity = np.std(pattern[mask])
                    radial_complexity.append(complexity)
            
            if len(radial_complexity) > 1:
                # Check if complexity increases with radius (spiral-like)
                correlation = np.corrcoef(range(len(radial_complexity)), radial_complexity)[0, 1]
                spiral_score = max(0, correlation)
        
        return {
            'has_spiral': spiral_score > 0.3,
            'spiral_score': spiral_score,
            'center': center
        }
    
    def analyze_topology(self, pattern: np.ndarray) -> Dict[str, Any]:
        """
        Analyze topological properties of the Kolam pattern.
        
        Args:
            pattern: 2D numpy array representing the Kolam pattern
            
        Returns:
            Dictionary containing topological analysis results
        """
        # Convert to binary
        binary_pattern = (pattern > 0).astype(int)
        
        # Find connected components
        from scipy.ndimage import label
        labeled_array, num_features = label(binary_pattern)
        
        # Analyze each connected component
        component_analysis = []
        for i in range(1, num_features + 1):
            component = (labeled_array == i)
            analysis = self._analyze_component_topology(component)
            component_analysis.append(analysis)
        
        # Overall topology
        total_components = num_features
        total_holes = sum(comp['holes'] for comp in component_analysis)
        total_loops = sum(comp['loops'] for comp in component_analysis)
        
        return {
            'total_components': total_components,
            'total_holes': total_holes,
            'total_loops': total_loops,
            'component_analysis': component_analysis,
            'is_continuous': total_components == 1,
            'has_loops': total_loops > 0
        }
    
    def _analyze_component_topology(self, component: np.ndarray) -> Dict[str, Any]:
        """Analyze topology of a single connected component."""
        # Count holes using Euler characteristic
        # Simplified approach - count enclosed regions
        
        # Find contours
        contours, _ = cv2.findContours(
            component.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Count holes (simplified)
        holes = 0
        if len(contours) > 0:
            # Use contour hierarchy to detect holes
            # This is a simplified implementation
            area = cv2.contourArea(contours[0])
            if area > 100:  # Threshold for significant holes
                holes = 1
        
        # Detect loops (closed curves)
        loops = 1 if len(contours) > 0 else 0
        
        return {
            'area': np.sum(component),
            'holes': holes,
            'loops': loops,
            'contours': len(contours)
        }
    
    def generate_analysis_report(self, pattern: np.ndarray) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report for a Kolam pattern.
        
        Args:
            pattern: 2D numpy array representing the Kolam pattern
            
        Returns:
            Dictionary containing complete analysis results
        """
        report = {
            'pattern_info': {
                'shape': pattern.shape,
                'dtype': str(pattern.dtype),
                'min_value': float(np.min(pattern)),
                'max_value': float(np.max(pattern)),
                'mean_value': float(np.mean(pattern))
            },
            'symmetry_analysis': self.analyze_symmetry(pattern),
            'fractal_analysis': self.analyze_fractal_properties(pattern),
            'topology_analysis': self.analyze_topology(pattern)
        }
        
        # Overall classification
        report['classification'] = self._classify_pattern(report)
        
        return report
    
    def _classify_pattern(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the pattern based on analysis results."""
        classification = {
            'primary_type': 'unknown',
            'secondary_types': [],
            'complexity': 'low',
            'mathematical_properties': []
        }
        
        # Determine primary type
        symmetry = analysis['symmetry_analysis']
        fractal = analysis['fractal_analysis']
        topology = analysis['topology_analysis']
        
        if symmetry['radial_symmetry']['is_radial']:
            classification['primary_type'] = 'radial'
            classification['mathematical_properties'].append('radial_symmetry')
        
        elif symmetry['bilateral_symmetry']['is_bilateral']:
            classification['primary_type'] = 'bilateral'
            classification['mathematical_properties'].append('bilateral_symmetry')
        
        elif fractal['is_fractal']:
            classification['primary_type'] = 'fractal'
            classification['mathematical_properties'].append('fractal_geometry')
        
        elif topology['is_continuous']:
            classification['primary_type'] = 'continuous'
            classification['mathematical_properties'].append('topology')
        
        # Add secondary types
        if fractal['is_fractal'] and classification['primary_type'] != 'fractal':
            classification['secondary_types'].append('fractal')
        
        if topology['has_loops']:
            classification['mathematical_properties'].append('closed_curves')
        
        # Determine complexity
        complexity_score = 0
        if fractal['box_dimension'] > 1.5:
            complexity_score += 1
        if topology['total_components'] > 1:
            complexity_score += 1
        if len(classification['mathematical_properties']) > 2:
            complexity_score += 1
        
        if complexity_score >= 2:
            classification['complexity'] = 'high'
        elif complexity_score == 1:
            classification['complexity'] = 'medium'
        
        return classification
    
    def save_analysis(self, analysis: Dict[str, Any], filename: str):
        """Save analysis results to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
    
    def visualize_analysis(self, pattern: np.ndarray, analysis: Dict[str, Any], save_path: str = None):
        """Visualize the analysis results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Kolam Pattern Analysis', fontsize=16)
        
        # Original pattern
        axes[0, 0].imshow(pattern, cmap='viridis')
        axes[0, 0].set_title('Original Pattern')
        axes[0, 0].axis('off')
        
        # Symmetry visualization
        symmetry = analysis['symmetry_analysis']
        if symmetry['radial_symmetry']['is_radial']:
            center = symmetry['radial_symmetry']['center']
            axes[0, 1].imshow(pattern, cmap='viridis')
            axes[0, 1].plot(center[1], center[0], 'r+', markersize=10)
            axes[0, 1].set_title('Radial Symmetry')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Radial Symmetry', ha='center', va='center')
            axes[0, 1].set_title('Radial Symmetry')
        axes[0, 1].axis('off')
        
        # Fractal visualization
        fractal = analysis['fractal_analysis']
        axes[0, 2].text(0.5, 0.5, f'Box Dimension: {fractal["box_dimension"]:.2f}', 
                       ha='center', va='center')
        axes[0, 2].set_title('Fractal Properties')
        axes[0, 2].axis('off')
        
        # Topology visualization
        topology = analysis['topology_analysis']
        axes[1, 0].text(0.5, 0.5, f'Components: {topology["total_components"]}\n'
                                 f'Holes: {topology["total_holes"]}\n'
                                 f'Loops: {topology["total_loops"]}', 
                       ha='center', va='center')
        axes[1, 0].set_title('Topology')
        axes[1, 0].axis('off')
        
        # Classification
        classification = analysis['classification']
        axes[1, 1].text(0.5, 0.5, f'Type: {classification["primary_type"]}\n'
                                 f'Complexity: {classification["complexity"]}', 
                       ha='center', va='center')
        axes[1, 1].set_title('Classification')
        axes[1, 1].axis('off')
        
        # Mathematical properties
        props = classification['mathematical_properties']
        axes[1, 2].text(0.5, 0.5, '\n'.join(props), ha='center', va='center')
        axes[1, 2].set_title('Mathematical Properties')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """Example usage of the KolamAnalyzer."""
    # Create a sample Kolam pattern for testing
    size = 200
    pattern = np.zeros((size, size))
    
    # Create a simple radial pattern
    center = (size // 2, size // 2)
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    theta = np.arctan2(y - center[0], x - center[1])
    
    # Create a spiral-like pattern
    for i in range(8):
        angle = i * np.pi / 4
        mask = (np.abs(theta - angle) < 0.2) & (r < 80) & (r > 20)
        pattern[mask] = 1
    
    # Add some fractal-like details
    for scale in [1, 2, 4]:
        for i in range(8 * scale):
            angle = i * np.pi / (4 * scale)
            mask = (np.abs(theta - angle) < 0.1 / scale) & (r < 60 / scale) & (r > 10 / scale)
            pattern[mask] = 1
    
    # Analyze the pattern
    analyzer = KolamAnalyzer()
    analysis = analyzer.generate_analysis_report(pattern)
    
    # Print results
    print("Kolam Pattern Analysis Results:")
    print("=" * 40)
    print(f"Pattern Shape: {analysis['pattern_info']['shape']}")
    print(f"Primary Type: {analysis['classification']['primary_type']}")
    print(f"Complexity: {analysis['classification']['complexity']}")
    print(f"Mathematical Properties: {analysis['classification']['mathematical_properties']}")
    
    # Visualize results
    analyzer.visualize_analysis(pattern, analysis, 'kolam_analysis.png')
    
    # Save analysis
    analyzer.save_analysis(analysis, 'kolam_analysis.json')


if __name__ == "__main__":
    main()
