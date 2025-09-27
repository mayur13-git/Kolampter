"""
Debug version of the Kolam Scanner web application.
This version includes additional logging and error handling.
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import base64
import io
import numpy as np
import cv2
from PIL import Image
import json
from datetime import datetime
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Import our Kolam analysis system
try:
    from kolam_analyzer import KolamAnalyzer
    from kolam_generator import KolamGenerator
    from kolam_visualizer import KolamVisualizer
    
    # Initialize Kolam analysis components
    analyzer = KolamAnalyzer()
    generator = KolamGenerator()
    visualizer = KolamVisualizer()
    logger.info("Kolam analysis modules loaded successfully")
except ImportError as e:
    logger.error(f"Failed to import Kolam modules: {e}")
    analyzer = None
    generator = None
    visualizer = None


@app.route('/')
def index():
    """Main page with camera and upload interface."""
    logger.info("Serving index page")
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        return f"Error loading page: {e}", 500


@app.route('/scan')
def scan():
    """Scan page for camera capture."""
    logger.info("Serving scan page")
    try:
        return render_template('scan.html')
    except Exception as e:
        logger.error(f"Error rendering scan: {e}")
        return f"Error loading scan page: {e}", 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'analyzer': analyzer is not None,
        'generator': generator is not None,
        'visualizer': visualizer is not None
    })


@app.route('/debug')
def debug():
    """Debug information endpoint."""
    return jsonify({
        'templates_dir': app.template_folder,
        'static_dir': app.static_folder,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'results_folder': app.config['RESULTS_FOLDER'],
        'templates_exist': os.path.exists('templates'),
        'static_exist': os.path.exists('static'),
        'index_template': os.path.exists('templates/index.html'),
        'scan_template': os.path.exists('templates/scan.html'),
        'base_template': os.path.exists('templates/base.html'),
        'static_css': os.path.exists('static/style.css')
    })


@app.route('/api/pattern-types')
def get_pattern_types():
    """Get available pattern types."""
    return jsonify({
        'pattern_types': ['fractal', 'spiral', 'grid', 'geometric', 'traditional'],
        'symmetry_types': ['radial', 'bilateral', 'rotational', 'none'],
        'color_schemes': ['monochrome', 'gradient', 'rainbow']
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    logger.error(f"404 error: {request.url}")
    return jsonify({
        'error': 'Not found',
        'url': request.url,
        'method': request.method
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"500 error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500


if __name__ == '__main__':
    print("🔧 Debug Kolam Scanner Web Application")
    print("=" * 50)
    print("Starting debug server...")
    print("Available endpoints:")
    print("  / - Home page")
    print("  /scan - Camera scan page")
    print("  /health - Health check")
    print("  /debug - Debug information")
    print("  /api/pattern-types - API endpoint")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
