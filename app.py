"""
Kolam Scanner Web Application
============================

A Flask web application that allows users to capture or upload Kolam images
and analyze them to identify design principles and mathematical patterns.
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import base64
import io
import numpy as np
import cv2
from PIL import Image
import json
from datetime import datetime
import uuid
import traceback

# Import our Kolam analysis system
from kolam_analyzer import KolamAnalyzer
from kolam_generator import KolamGenerator
from kolam_visualizer import KolamVisualizer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
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


def json_serializer(obj):
    """Custom JSON serializer for numpy types and other non-serializable objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    else:
        return str(obj)


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
        
        # Resize if too large (for faster performance)
        height, width = gray.shape
        if max(height, width) > 500:  # Reduced from 1000 to 500 for faster processing
            scale = 500 / max(height, width)
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
        # Perform quick analysis (skip some complex calculations for speed)
        analysis = analyzer.generate_analysis_report(image_array)
        
        # Simplify some analysis results for faster processing
        if 'fractal_analysis' in analysis:
            # Keep only essential fractal data
            fractal = analysis['fractal_analysis']
            analysis['fractal_analysis'] = {
                'box_dimension': fractal.get('box_dimension', 1.0),
                'self_similarity': fractal.get('self_similarity', 0.0),
                'is_fractal': fractal.get('is_fractal', False)
            }
        
        # Generate visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = str(uuid.uuid4())[:8]
        
        # Create result directory for this session
        result_dir = os.path.join(app.config['RESULTS_FOLDER'], f"session_{session_id}_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # Save original image
        original_path = os.path.join(result_dir, "original.png")
        cv2.imwrite(original_path, image_array)
        
        # Create analysis visualizations (faster version)
        analysis_path = os.path.join(result_dir, "analysis.png")
        try:
            # Use the global visualizer object
            # Fallback for missing 'recursive_structure' in fractal_analysis
            if 'fractal_analysis' in analysis and 'recursive_structure' not in analysis['fractal_analysis']:
                analysis['fractal_analysis']['recursive_structure'] = {'has_recursive_structure': 0.0}
            visualizer.plot_analysis_heatmap(analysis, analysis_path)
        except Exception as e:
            print(f"Analysis heatmap failed: {e}")
            analysis_path = None
        
        # Create symmetry visualization (simplified)
        symmetry_path = os.path.join(result_dir, "symmetry.png")
        try:
            # Create a simple symmetry visualization instead of the full one
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(image_array, cmap='viridis')
            ax.set_title('Symmetry Analysis')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(symmetry_path, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Symmetry visualization failed: {e}")
            symmetry_path = None
        
        # Always generate fractal analysis visualization
        fractal_path = os.path.join(result_dir, "fractal.png")
        try:
            # Use the global visualizer object
            visualizer.plot_pattern(image_array, title="Fractal Analysis", save_path=fractal_path)
        except Exception as e:
            print(f"Fractal visualization failed: {e}")
            fractal_path = None
        
        # Always generate 3D visualization
        visualizer_3d_path = os.path.join(result_dir, "3d_visualization.png")
        try:
            visualizer.plot_3d_pattern(image_array, title="3D Kolam Pattern", save_path=visualizer_3d_path)
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
            json.dump(analysis, f, indent=2, default=json_serializer)
        
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
        print("Upload request received")
        
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"Processing file: {file.filename}")
        
        if file:
            # Save uploaded file
            filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved to: {filepath}")
            
            # Preprocess image
            print("Preprocessing image...")
            processed_image = preprocess_image(filepath)
            if processed_image is None:
                print("Image preprocessing failed")
                return jsonify({'error': 'Failed to process image'}), 400
            
            print(f"Image preprocessed, shape: {processed_image.shape}")
            
            # Analyze the image
            print("Starting analysis...")
            analysis_results = analyze_kolam_image(processed_image)
            print("Analysis completed")
            
            if 'error' in analysis_results:
                print(f"Analysis error: {analysis_results['error']}")
                return jsonify({'error': analysis_results['error']}), 400
            
            # Convert numpy types to JSON-serializable types
            print("Serializing results...")
            serializable_results = json.loads(json.dumps(analysis_results, default=json_serializer))
            print("Results serialized successfully")
            return jsonify(serializable_results)
    
    except Exception as e:
        print(f"Upload error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


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
        
        # Convert numpy types to JSON-serializable types
        serializable_results = json.loads(json.dumps(analysis_results, default=json_serializer))
        return jsonify(serializable_results)
    
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
        
        result = {
            'session_id': session_id,
            'pattern_type': pattern_type,
            'analysis': analysis,
            'image_path': generated_path
        }
        
        # Convert numpy types to JSON-serializable types
        serializable_result = json.loads(json.dumps(result, default=json_serializer))
        return jsonify(serializable_result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Kolam Scanner is running',
        'timestamp': datetime.now().isoformat()
    })

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
