"""
Test script for the Kolam Scanner web application.
"""

import os
import sys
import numpy as np
from PIL import Image
import io
import base64

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from app import app, preprocess_image, analyze_kolam_image
        print("✓ Flask app imports successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    try:
        from kolam_analyzer import KolamAnalyzer
        from kolam_generator import KolamGenerator
        from kolam_visualizer import KolamVisualizer
        print("✓ Kolam analysis modules import successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True

def test_image_processing():
    """Test image processing functionality."""
    print("\nTesting image processing...")
    
    try:
        from app import preprocess_image
        
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        data_url = f"data:image/png;base64,{img_str}"
        
        # Test preprocessing
        processed = preprocess_image(data_url)
        
        if processed is not None and processed.shape[0] > 0:
            print("✓ Image preprocessing works")
            return True
        else:
            print("✗ Image preprocessing failed")
            return False
            
    except Exception as e:
        print(f"✗ Image processing error: {e}")
        return False

def test_analysis():
    """Test analysis functionality."""
    print("\nTesting analysis...")
    
    try:
        from app import analyze_kolam_image
        import cv2
        
        # Create a simple test pattern
        test_pattern = np.zeros((100, 100), dtype=np.uint8)
        
        # Add some simple geometric shapes
        cv2.circle(test_pattern, (50, 50), 30, 255, 2)
        cv2.rectangle(test_pattern, (20, 20), (80, 80), 255, 2)
        
        # Test analysis
        results = analyze_kolam_image(test_pattern)
        
        if 'error' not in results and 'classification' in results:
            print("✓ Analysis works")
            print(f"  - Pattern type: {results['classification']['primary_type']}")
            print(f"  - Complexity: {results['classification']['complexity']}")
            return True
        else:
            print("✗ Analysis failed")
            return False
            
    except Exception as e:
        print(f"✗ Analysis error: {e}")
        return False

def test_flask_app():
    """Test Flask app creation."""
    print("\nTesting Flask app...")
    
    try:
        from app import app
        
        # Test app creation
        with app.test_client() as client:
            # Test home page
            response = client.get('/')
            if response.status_code == 200:
                print("✓ Home page loads successfully")
            else:
                print(f"✗ Home page failed: {response.status_code}")
                return False
            
            # Test scan page
            response = client.get('/scan')
            if response.status_code == 200:
                print("✓ Scan page loads successfully")
            else:
                print(f"✗ Scan page failed: {response.status_code}")
                return False
            
            # Test API endpoint
            response = client.get('/api/pattern-types')
            if response.status_code == 200:
                print("✓ API endpoint works")
            else:
                print(f"✗ API endpoint failed: {response.status_code}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Flask app error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Kolam Scanner Web App Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_image_processing,
        test_analysis,
        test_flask_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Web app is ready to use.")
        print("\nTo start the web application:")
        print("  python run_web_app.py")
        print("\nThen open: http://localhost:5000")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
