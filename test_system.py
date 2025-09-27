"""
Simple test script to verify the Kolam system works without optional dependencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from kolam_analyzer import KolamAnalyzer
from kolam_generator import KolamGenerator, KolamParameters
from kolam_visualizer import KolamVisualizer


def test_basic_functionality():
    """Test basic functionality without optional dependencies."""
    print("Testing Kolam System...")
    print("=" * 40)
    
    try:
        # Test generator
        print("1. Testing pattern generation...")
        generator = KolamGenerator()
        pattern = generator.generate('fractal', size=100, complexity=2)
        print(f"   ✓ Generated pattern with shape: {pattern.shape}")
        
        # Test analyzer
        print("2. Testing pattern analysis...")
        analyzer = KolamAnalyzer()
        analysis = analyzer.generate_analysis_report(pattern)
        print(f"   ✓ Analysis completed. Pattern type: {analysis['classification']['primary_type']}")
        
        # Test visualizer
        print("3. Testing visualization...")
        visualizer = KolamVisualizer()
        visualizer.plot_pattern(pattern, "Test Pattern", "test_pattern.png")
        print("   ✓ Visualization completed")
        
        # Test different pattern types
        print("4. Testing different pattern types...")
        pattern_types = ['spiral', 'geometric', 'traditional']
        for pt in pattern_types:
            test_pattern = generator.generate(pt, size=50, complexity=1)
            print(f"   ✓ {pt} pattern generated")
        
        print("\n🎉 All tests passed! System is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\nYou can now run:")
        print("  python demo.py")
        print("  python examples.py")
    else:
        print("\nPlease check the error and try again.")
