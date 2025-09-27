"""
Kolam System Demo
================

A simple demonstration script showing the main features of the Kolam
analysis and generation system.
"""

import numpy as np
import matplotlib.pyplot as plt
from kolam_analyzer import KolamAnalyzer
from kolam_generator import KolamGenerator, KolamParameters
from kolam_visualizer import KolamVisualizer


def main():
    """Run a simple demonstration of the Kolam system."""
    print("🎨 Kolam Design Analysis and Generation System Demo")
    print("=" * 60)
    
    # Initialize components
    generator = KolamGenerator()
    analyzer = KolamAnalyzer()
    visualizer = KolamVisualizer()
    
    print("\n1. Generating Kolam Patterns...")
    print("-" * 40)
    
    # Generate different types of patterns
    pattern_types = ['fractal', 'spiral', 'geometric', 'traditional']
    patterns = []
    
    for pattern_type in pattern_types:
        print(f"   Generating {pattern_type} pattern...")
        pattern = generator.generate(pattern_type, size=200, complexity=3)
        patterns.append(pattern)
    
    print("   ✓ All patterns generated successfully!")
    
    print("\n2. Analyzing Patterns...")
    print("-" * 40)
    
    # Analyze each pattern
    for i, (pattern, pattern_type) in enumerate(zip(patterns, pattern_types)):
        print(f"   Analyzing {pattern_type} pattern...")
        analysis = analyzer.generate_analysis_report(pattern)
        
        classification = analysis['classification']
        print(f"   - Type: {classification['primary_type']}")
        print(f"   - Complexity: {classification['complexity']}")
        print(f"   - Properties: {', '.join(classification['mathematical_properties'])}")
    
    print("   ✓ All patterns analyzed successfully!")
    
    print("\n3. Creating Visualizations...")
    print("-" * 40)
    
    # Create visualizations
    print("   Creating pattern comparison...")
    visualizer.plot_comparison(patterns, pattern_types, 'demo_comparison.png')
    
    print("   Creating 3D visualization...")
    visualizer.plot_3d_pattern(patterns[0], "3D Fractal Kolam", 'demo_3d.png')
    
    print("   Creating analysis visualization...")
    analysis = analyzer.generate_analysis_report(patterns[0])
    visualizer.plot_analysis_heatmap(analysis, 'demo_analysis.png')
    
    print("   ✓ All visualizations created successfully!")
    
    print("\n4. Interactive Example...")
    print("-" * 40)
    
    # Create an interactive example
    print("   Creating custom fractal pattern...")
    custom_params = KolamParameters(
        size=300,
        complexity=4,
        symmetry_type='radial',
        pattern_type='fractal',
        color_scheme='gradient'
    )
    
    # Remove pattern_type from kwargs to avoid conflict
    custom_kwargs = {k: v for k, v in custom_params.__dict__.items() if k != 'pattern_type'}
    custom_pattern = generator.generate('fractal', **custom_kwargs)
    custom_analysis = analyzer.generate_analysis_report(custom_pattern)
    
    print(f"   - Pattern size: {custom_pattern.shape}")
    print(f"   - Fractal dimension: {custom_analysis['fractal_analysis']['box_dimension']:.3f}")
    print(f"   - Radial symmetry: {custom_analysis['symmetry_analysis']['radial_symmetry']['is_radial']}")
    
    visualizer.plot_pattern(custom_pattern, "Custom Fractal Kolam", 'demo_custom.png')
    
    print("   ✓ Custom example created successfully!")
    
    print("\n5. Summary...")
    print("-" * 40)
    
    print("   Generated files:")
    print("   - demo_comparison.png (Pattern comparison)")
    print("   - demo_3d.png (3D visualization)")
    print("   - demo_analysis.png (Analysis heatmap)")
    print("   - demo_custom.png (Custom pattern)")
    
    print("\n   System capabilities demonstrated:")
    print("   ✓ Pattern generation (fractal, spiral, geometric, traditional)")
    print("   ✓ Mathematical analysis (symmetry, fractals, topology)")
    print("   ✓ 2D and 3D visualization")
    print("   ✓ Analysis visualization")
    print("   ✓ Custom parameter configuration")
    
    print("\n🎉 Demo completed successfully!")
    print("\nTo explore more features, run:")
    print("   python examples.py")
    print("\nFor interactive turtle graphics, run:")
    print("   python kolam_generator.py")


if __name__ == "__main__":
    main()
