"""
Kolam Scanner Web Application
============================

A Flask web application that allows users to capture or upload Kolam images
and analyze them to identify design principles and mathematical patterns.
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import base64
import io
import numpy as np
import cv2
from PIL import Image
import json
from datetime import datetime
import uuid

# Import our Kolam analysis system
from kolam_analyzer import KolamAnalyzer
from kolam_generator import KolamGenerator
from kolam_visualizer import KolamVisualizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize Kolam analysis components
analyzer = KolamAnalyzer()
generator = KolamGenerator()
visualizer = KolamVisualizer()


def preprocess_image(image_data):
    """
    Preprocess uploaded image for Kolam analysis.
    
    Args:
        image_data: Base64 encoded image data or file path
        
    Returns:
        Preprocessed numpy array suitable for analysis
    """
    try:
        # Handle base64 data (from camera capture)
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Remove data URL prefix
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            # Handle file path
            image = Image.open(image_data)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Resize if too large (for performance)
        height, width = gray.shape
        if max(height, width) > 1000:
            scale = 1000 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height))
        
        # Apply image enhancement
        # Convert to binary (black and white) for better analysis
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Invert if necessary (make lines white on black background)
        if np.mean(binary) > 127:
            binary = 255 - binary
        
        return binary
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def analyze_kolam_image(image_array):
    """
    Analyze a Kolam image and return comprehensive results.
    
    Args:
        image_array: Preprocessed numpy array of the Kolam image
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        # Perform comprehensive analysis
        analysis = analyzer.generate_analysis_report(image_array)
        
        # Generate visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = str(uuid.uuid4())[:8]
        
        # Create result directory for this session
        result_dir = os.path.join(app.config['RESULTS_FOLDER'], f"session_{session_id}_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # Save original image
        original_path = os.path.join(result_dir, "original.png")
        cv2.imwrite(original_path, image_array)
        
        # Create analysis visualizations
        analysis_path = os.path.join(result_dir, "analysis.png")
        visualizer.plot_analysis_heatmap(analysis, analysis_path)
        
        # Create symmetry visualization
        symmetry_path = os.path.join(result_dir, "symmetry.png")
        visualizer.create_symmetry_visualization(image_array, symmetry_path)
        
        # Create fractal analysis
        fractal_path = os.path.join(result_dir, "fractal.png")
        visualizer.plot_fractal_analysis(image_array, fractal_path)
        
        # Create 3D visualization
        try:
            visualizer_3d_path = os.path.join(result_dir, "3d_visualization.png")
            visualizer.plot_3d_pattern(image_array, "3D Kolam Analysis", visualizer_3d_path)
        except Exception as e:
            print(f"3D visualization failed: {e}")
            visualizer_3d_path = None
        
        # Add file paths to analysis results
        analysis['visualization_paths'] = {
            'original': original_path,
            'analysis': analysis_path,
            'symmetry': symmetry_path,
            'fractal': fractal_path,
            '3d': visualizer_3d_path
        }
        
        # Save analysis results as JSON
        results_path = os.path.join(result_dir, "analysis_results.json")
        with open(results_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        analysis['session_id'] = session_id
        analysis['timestamp'] = timestamp
        analysis['results_path'] = results_path
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return {'error': str(e)}


@app.route('/')
def index():
    """Main page with camera and upload interface."""
    return render_template('index.html')


@app.route('/scan')
def scan():
    """Scan page for camera capture."""
    return render_template('scan.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            # Save uploaded file
            filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess image
            processed_image = preprocess_image(filepath)
            if processed_image is None:
                return jsonify({'error': 'Failed to process image'}), 400
            
            # Analyze the image
            analysis_results = analyze_kolam_image(processed_image)
            
            if 'error' in analysis_results:
                return jsonify({'error': analysis_results['error']}), 400
            
            return jsonify(analysis_results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/capture', methods=['POST'])
def capture_image():
    """Handle camera capture and analysis."""
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data received'}), 400
        
        image_data = data['image']
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Analyze the image
        analysis_results = analyze_kolam_image(processed_image)
        
        if 'error' in analysis_results:
            return jsonify({'error': analysis_results['error']}), 400
        
        return jsonify(analysis_results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/results/<session_id>')
def view_results(session_id):
    """View analysis results for a specific session."""
    # Find the result directory
    result_dirs = [d for d in os.listdir(app.config['RESULTS_FOLDER']) 
                   if d.startswith(f"session_{session_id}")]
    
    if not result_dirs:
        return jsonify({'error': 'Results not found'}), 404
    
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], result_dirs[0])
    results_file = os.path.join(result_dir, "analysis_results.json")
    
    if not os.path.exists(results_file):
        return jsonify({'error': 'Analysis results not found'}), 404
    
    with open(results_file, 'r') as f:
        analysis_results = json.load(f)
    
    return jsonify(analysis_results)


@app.route('/image/<session_id>/<image_type>')
def get_image(session_id, image_type):
    """Serve analysis images."""
    # Find the result directory
    result_dirs = [d for d in os.listdir(app.config['RESULTS_FOLDER']) 
                   if d.startswith(f"session_{session_id}")]
    
    if not result_dirs:
        return jsonify({'error': 'Results not found'}), 404
    
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], result_dirs[0])
    
    image_map = {
        'original': 'original.png',
        'analysis': 'analysis.png',
        'symmetry': 'symmetry.png',
        'fractal': 'fractal.png',
        '3d': '3d_visualization.png'
    }
    
    if image_type not in image_map:
        return jsonify({'error': 'Invalid image type'}), 400
    
    image_path = os.path.join(result_dir, image_map[image_type])
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    return send_file(image_path, mimetype='image/png')


@app.route('/generate', methods=['POST'])
def generate_kolam():
    """Generate a new Kolam pattern based on analysis results."""
    try:
        data = request.get_json()
        pattern_type = data.get('pattern_type', 'fractal')
        size = data.get('size', 300)
        complexity = data.get('complexity', 3)
        symmetry_type = data.get('symmetry_type', 'radial')
        
        # Generate pattern
        pattern = generator.generate(
            pattern_type=pattern_type,
            size=size,
            complexity=complexity,
            symmetry_type=symmetry_type
        )
        
        # Save generated pattern
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = str(uuid.uuid4())[:8]
        
        result_dir = os.path.join(app.config['RESULTS_FOLDER'], f"generated_{session_id}_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        generated_path = os.path.join(result_dir, "generated_kolam.png")
        generator.save_pattern(pattern, generated_path)
        
        # Analyze the generated pattern
        analysis = analyzer.generate_analysis_report(pattern)
        
        return jsonify({
            'session_id': session_id,
            'pattern_type': pattern_type,
            'analysis': analysis,
            'image_path': generated_path
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pattern-types')
def get_pattern_types():
    """Get available pattern types."""
    return jsonify({
        'pattern_types': ['fractal', 'spiral', 'grid', 'geometric', 'traditional'],
        'symmetry_types': ['radial', 'bilateral', 'rotational', 'none'],
        'color_schemes': ['monochrome', 'gradient', 'rainbow']
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
