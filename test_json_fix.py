"""
Test script to verify the JSON serialization fix works.
"""

import numpy as np
import json
from app import app, json_serializer, analyze_kolam_image

def test_json_serialization():
    """Test JSON serialization with various data types."""
    print("Testing JSON serialization...")
    
    # Test data with various types that cause issues
    test_data = {
        'bool_val': True,
        'numpy_bool': np.bool_(True),
        'numpy_int': np.int32(42),
        'numpy_float': np.float64(3.14),
        'numpy_array': np.array([1, 2, 3]),
        'nested': {
            'inner_bool': np.bool_(False),
            'inner_float': np.float32(2.71)
        }
    }
    
    try:
        # Test serialization
        json_str = json.dumps(test_data, default=json_serializer)
        print("✓ JSON serialization successful")
        
        # Test deserialization
        parsed_data = json.loads(json_str)
        print("✓ JSON deserialization successful")
        
        return True
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
        return False

def test_analysis_serialization():
    """Test that analysis results can be serialized."""
    print("\nTesting analysis serialization...")
    
    try:
        # Create a simple test pattern
        test_pattern = np.zeros((50, 50), dtype=np.uint8)
        test_pattern[20:30, 20:30] = 255  # Simple square
        
        # Analyze the pattern
        analysis = analyze_kolam_image(test_pattern)
        
        if 'error' in analysis:
            print(f"✗ Analysis failed: {analysis['error']}")
            return False
        
        # Test JSON serialization of analysis results
        json_str = json.dumps(analysis, default=json_serializer)
        print("✓ Analysis results serialization successful")
        
        # Test deserialization
        parsed_analysis = json.loads(json_str)
        print("✓ Analysis results deserialization successful")
        
        return True
    except Exception as e:
        print(f"✗ Analysis serialization failed: {e}")
        return False

def test_flask_endpoints():
    """Test Flask endpoints with JSON responses."""
    print("\nTesting Flask endpoints...")
    
    try:
        with app.test_client() as client:
            # Test API endpoint
            response = client.get('/api/pattern-types')
            if response.status_code == 200:
                print("✓ API endpoint works")
            else:
                print(f"✗ API endpoint failed: {response.status_code}")
                return False
            
            # Test home page
            response = client.get('/')
            if response.status_code == 200:
                print("✓ Home page loads")
            else:
                print(f"✗ Home page failed: {response.status_code}")
                return False
            
            return True
    except Exception as e:
        print(f"✗ Flask endpoint test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 JSON Serialization Fix Test Suite")
    print("=" * 40)
    
    tests = [
        test_json_serialization,
        test_analysis_serialization,
        test_flask_endpoints
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
        print("🎉 All tests passed! JSON serialization is fixed.")
        print("\nThe web application should now work without JSON errors.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
