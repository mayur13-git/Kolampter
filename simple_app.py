"""
Simple test version of the Kolam Scanner web application for debugging.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import traceback

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check."""
    return jsonify({
        'status': 'healthy',
        'message': 'Simple app is running'
    })

@app.route('/upload', methods=['POST'])
def upload():
    """Simple upload test."""
    try:
        print("Upload request received")
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"File received: {file.filename}")
        
        # Return a mock analysis result that matches the expected format
        return jsonify({
            'session_id': 'test123',
            'classification': {
                'primary_type': 'fractal',
                'complexity': 'medium',
                'mathematical_properties': ['radial_symmetry', 'fractal_geometry']
            },
            'symmetry_analysis': {
                'radial_symmetry': {
                    'is_radial': True,
                    'score': 0.85
                },
                'bilateral_symmetry': {
                    'is_bilateral': False,
                    'max_score': 0.3
                },
                'rotational_symmetry': {
                    'is_rotational': True,
                    'best_score': 0.7
                }
            },
            'fractal_analysis': {
                'box_dimension': 1.6,
                'self_similarity': 0.75,
                'is_fractal': True
            },
            'topology_analysis': {
                'total_components': 1,
                'total_holes': 3,
                'total_loops': 2
            },
            'visualization_paths': {
                'original': '/test/original.png',
                'analysis': '/test/analysis.png',
                'symmetry': '/test/symmetry.png',
                'fractal': None,
                '3d': None
            }
        })
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/image/<session_id>/<image_type>')
def get_mock_image(session_id, image_type):
    """Return a mock image response."""
    # Create a simple 1x1 pixel image
    from PIL import Image
    import io
    import base64
    
    # Create a simple colored square
    img = Image.new('RGB', (100, 100), color='lightblue')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

if __name__ == '__main__':
    print("🔧 Simple Test App Starting...")
    print("Available endpoints:")
    print("  / - Home page")
    print("  /health - Health check")
    print("  /upload - File upload test")
    app.run(debug=True, host='0.0.0.0', port=5000)
