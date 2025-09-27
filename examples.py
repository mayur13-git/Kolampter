"""
Kolam Examples and Test Cases
=============================

This module provides comprehensive examples and test cases for the Kolam
analysis and generation system.
"""

import numpy as np
import matplotlib.pyplot as plt
from kolam_analyzer import KolamAnalyzer
from kolam_generator import KolamGenerator, KolamParameters, TurtleKolamGenerator
from kolam_visualizer import KolamVisualizer
import json
import os
from typing import List, Dict, Any


class KolamExamples:
    """Collection of example Kolam patterns and test cases."""
    
    def __init__(self):
        self.analyzer = KolamAnalyzer()
        self.generator = KolamGenerator()
        self.visualizer = KolamVisualizer()
        self.output_dir = "kolam_examples"
        self._create_output_directory()
    
    def _create_output_directory(self):
        """Create output directory for examples."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def create_basic_examples(self):
        """Create basic Kolam pattern examples."""
        print("Creating basic Kolam examples...")
        
        # Example 1: Simple radial pattern
        self._create_radial_example()
        
        # Example 2: Fractal pattern
        self._create_fractal_example()
        
        # Example 3: Spiral pattern
        self._create_spiral_example()
        
        # Example 4: Grid pattern
        self._create_grid_example()
        
        # Example 5: Traditional pattern
        self._create_traditional_example()
    
    def _create_radial_example(self):
        """Create a radial symmetry example."""
        # Generate radial pattern
        params = KolamParameters(
            size=200,
            complexity=3,
            symmetry_type='radial',
            pattern_type='geometric'
        )
        
        pattern = self.generator.generate('geometric', **params.__dict__)
        
        # Analyze the pattern
        analysis = self.analyzer.generate_analysis_report(pattern)
        
        # Visualize
        self.visualizer.plot_pattern(pattern, "Radial Symmetry Example", 
                                   f"{self.output_dir}/radial_example.png")
        
        # Save analysis
        with open(f"{self.output_dir}/radial_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print("✓ Radial symmetry example created")
    
    def _create_fractal_example(self):
        """Create a fractal pattern example."""
        # Generate fractal pattern
        params = KolamParameters(
            size=300,
            complexity=4,
            symmetry_type='none',
            pattern_type='fractal'
        )
        
        pattern = self.generator.generate('fractal', **params.__dict__)
        
        # Analyze the pattern
        analysis = self.analyzer.generate_analysis_report(pattern)
        
        # Visualize
        self.visualizer.plot_pattern(pattern, "Fractal Pattern Example", 
                                   f"{self.output_dir}/fractal_example.png")
        
        # Create detailed fractal analysis
        self.visualizer.plot_fractal_analysis(pattern, 
                                            f"{self.output_dir}/fractal_analysis.png")
        
        # Save analysis
        with open(f"{self.output_dir}/fractal_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print("✓ Fractal pattern example created")
    
    def _create_spiral_example(self):
        """Create a spiral pattern example."""
        # Generate spiral pattern
        params = KolamParameters(
            size=250,
            complexity=3,
            symmetry_type='radial',
            pattern_type='spiral'
        )
        
        pattern = self.generator.generate('spiral', **params.__dict__)
        
        # Analyze the pattern
        analysis = self.analyzer.generate_analysis_report(pattern)
        
        # Visualize
        self.visualizer.plot_pattern(pattern, "Spiral Pattern Example", 
                                   f"{self.output_dir}/spiral_example.png")
        
        # Save analysis
        with open(f"{self.output_dir}/spiral_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print("✓ Spiral pattern example created")
    
    def _create_grid_example(self):
        """Create a grid pattern example."""
        # Generate grid pattern
        params = KolamParameters(
            size=200,
            complexity=2,
            symmetry_type='bilateral',
            pattern_type='grid'
        )
        
        pattern = self.generator.generate('grid', **params.__dict__)
        
        # Analyze the pattern
        analysis = self.analyzer.generate_analysis_report(pattern)
        
        # Visualize
        self.visualizer.plot_pattern(pattern, "Grid Pattern Example", 
                                   f"{self.output_dir}/grid_example.png")
        
        # Save analysis
        with open(f"{self.output_dir}/grid_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print("✓ Grid pattern example created")
    
    def _create_traditional_example(self):
        """Create a traditional Kolam pattern example."""
        # Generate traditional pattern
        params = KolamParameters(
            size=300,
            complexity=3,
            symmetry_type='radial',
            pattern_type='traditional'
        )
        
        pattern = self.generator.generate('traditional', **params.__dict__)
        
        # Analyze the pattern
        analysis = self.analyzer.generate_analysis_report(pattern)
        
        # Visualize
        self.visualizer.plot_pattern(pattern, "Traditional Kolam Example", 
                                   f"{self.output_dir}/traditional_example.png")
        
        # Save analysis
        with open(f"{self.output_dir}/traditional_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print("✓ Traditional pattern example created")
    
    def create_comparison_examples(self):
        """Create comparison examples showing different pattern types."""
        print("Creating comparison examples...")
        
        # Generate patterns of different types
        pattern_types = ['fractal', 'spiral', 'grid', 'geometric', 'traditional']
        patterns = []
        titles = []
        
        for pattern_type in pattern_types:
            pattern = self.generator.generate(pattern_type)
            patterns.append(pattern)
            titles.append(f"{pattern_type.title()} Pattern")
        
        # Create comparison visualization
        self.visualizer.plot_comparison(patterns, titles, 
                                      f"{self.output_dir}/pattern_comparison.png")
        
        # Create 3D comparison
        self.visualizer.plot_3d_pattern(patterns[0], "3D Fractal Kolam", 
                                      f"{self.output_dir}/3d_fractal.png")
        
        print("✓ Comparison examples created")
    
    def create_analysis_examples(self):
        """Create examples showing different analysis capabilities."""
        print("Creating analysis examples...")
        
        # Generate a complex pattern for analysis
        params = KolamParameters(
            size=300,
            complexity=4,
            symmetry_type='radial',
            pattern_type='fractal'
        )
        
        pattern = self.generator.generate('fractal', **params.__dict__)
        
        # Comprehensive analysis
        analysis = self.analyzer.generate_analysis_report(pattern)
        
        # Create analysis visualizations
        self.visualizer.plot_analysis_heatmap(analysis, 
                                            f"{self.output_dir}/analysis_heatmap.png")
        
        self.visualizer.create_symmetry_visualization(pattern, 
                                                    f"{self.output_dir}/symmetry_analysis.png")
        
        self.visualizer.plot_fractal_analysis(pattern, 
                                            f"{self.output_dir}/fractal_analysis_detailed.png")
        
        # Save comprehensive analysis
        with open(f"{self.output_dir}/comprehensive_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print("✓ Analysis examples created")
    
    def create_animation_examples(self):
        """Create animation examples."""
        print("Creating animation examples...")
        
        # Pattern evolution animation
        patterns = []
        for complexity in range(1, 6):
            params = KolamParameters(
                size=200,
                complexity=complexity,
                pattern_type='fractal'
            )
            pattern = self.generator.generate('fractal', **params.__dict__)
            patterns.append(pattern)
        
        # Create evolution animation
        self.visualizer.create_animation(patterns, "Fractal Pattern Evolution", 
                                       f"{self.output_dir}/fractal_evolution.gif")
        
        # Different pattern types animation
        pattern_types = ['fractal', 'spiral', 'geometric', 'traditional']
        patterns = [self.generator.generate(pt) for pt in pattern_types]
        
        self.visualizer.create_animation(patterns, "Kolam Pattern Types", 
                                       f"{self.output_dir}/pattern_types.gif")
        
        print("✓ Animation examples created")
    
    def create_turtle_examples(self):
        """Create interactive Turtle graphics examples."""
        print("Creating Turtle graphics examples...")
        
        # Fractal kolam
        turtle_gen = TurtleKolamGenerator(400)
        turtle_gen.draw_fractal_kolam(3)
        turtle_gen.save_screen(f"{self.output_dir}/turtle_fractal.eps")
        turtle_gen.close()
        
        # Spiral kolam
        turtle_gen = TurtleKolamGenerator(400)
        turtle_gen.draw_spiral_kolam(4)
        turtle_gen.save_screen(f"{self.output_dir}/turtle_spiral.eps")
        turtle_gen.close()
        
        # Geometric kolam
        turtle_gen = TurtleKolamGenerator(400)
        turtle_gen.draw_geometric_kolam(8)
        turtle_gen.save_screen(f"{self.output_dir}/turtle_geometric.eps")
        turtle_gen.close()
        
        print("✓ Turtle graphics examples created")
    
    def create_custom_examples(self):
        """Create custom Kolam pattern examples."""
        print("Creating custom examples...")
        
        # Custom fractal with specific parameters
        custom_params = KolamParameters(
            size=400,
            complexity=5,
            symmetry_type='radial',
            pattern_type='fractal',
            color_scheme='gradient'
        )
        
        pattern = self.generator.generate('fractal', **custom_params.__dict__)
        self.visualizer.plot_pattern(pattern, "Custom Fractal Kolam", 
                                   f"{self.output_dir}/custom_fractal.png")
        
        # Custom spiral with multiple spirals
        custom_params.pattern_type = 'spiral'
        custom_params.complexity = 6
        pattern = self.generator.generate('spiral', **custom_params.__dict__)
        self.visualizer.plot_pattern(pattern, "Custom Spiral Kolam", 
                                   f"{self.output_dir}/custom_spiral.png")
        
        # Custom geometric with high complexity
        custom_params.pattern_type = 'geometric'
        custom_params.complexity = 4
        pattern = self.generator.generate('geometric', **custom_params.__dict__)
        self.visualizer.plot_pattern(pattern, "Custom Geometric Kolam", 
                                   f"{self.output_dir}/custom_geometric.png")
        
        print("✓ Custom examples created")
    
    def run_comprehensive_test(self):
        """Run a comprehensive test of all functionality."""
        print("Running comprehensive test...")
        
        test_results = {
            'basic_generation': False,
            'analysis': False,
            'visualization': False,
            'turtle_graphics': False,
            'custom_patterns': False
        }
        
        try:
            # Test basic generation
            pattern = self.generator.generate('fractal')
            test_results['basic_generation'] = True
            print("✓ Basic generation test passed")
            
            # Test analysis
            analysis = self.analyzer.generate_analysis_report(pattern)
            test_results['analysis'] = True
            print("✓ Analysis test passed")
            
            # Test visualization
            self.visualizer.plot_pattern(pattern, "Test Pattern")
            test_results['visualization'] = True
            print("✓ Visualization test passed")
            
            # Test turtle graphics
            turtle_gen = TurtleKolamGenerator(200)
            turtle_gen.draw_fractal_kolam(2)
            turtle_gen.close()
            test_results['turtle_graphics'] = True
            print("✓ Turtle graphics test passed")
            
            # Test custom patterns
            custom_pattern = self.generator.generate('spiral', complexity=3, size=150)
            test_results['custom_patterns'] = True
            print("✓ Custom patterns test passed")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
        
        # Save test results
        with open(f"{self.output_dir}/test_results.json", 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"Test results: {test_results}")
        return test_results
    
    def create_documentation_examples(self):
        """Create examples for documentation."""
        print("Creating documentation examples...")
        
        # Create a simple example for beginners
        simple_params = KolamParameters(
            size=100,
            complexity=2,
            pattern_type='geometric'
        )
        
        simple_pattern = self.generator.generate('geometric', **simple_params.__dict__)
        self.visualizer.plot_pattern(simple_pattern, "Simple Kolam Example", 
                                   f"{self.output_dir}/simple_example.png")
        
        # Create a complex example for advanced users
        complex_params = KolamParameters(
            size=500,
            complexity=6,
            symmetry_type='radial',
            pattern_type='fractal',
            color_scheme='gradient'
        )
        
        complex_pattern = self.generator.generate('fractal', **complex_params.__dict__)
        self.visualizer.plot_pattern(complex_pattern, "Complex Kolam Example", 
                                   f"{self.output_dir}/complex_example.png")
        
        # Create analysis example
        analysis = self.analyzer.generate_analysis_report(complex_pattern)
        self.visualizer.plot_analysis_heatmap(analysis, 
                                            f"{self.output_dir}/documentation_analysis.png")
        
        print("✓ Documentation examples created")
    
    def run_all_examples(self):
        """Run all example creation functions."""
        print("Creating all Kolam examples...")
        print("=" * 50)
        
        self.create_basic_examples()
        print()
        
        self.create_comparison_examples()
        print()
        
        self.create_analysis_examples()
        print()
        
        self.create_animation_examples()
        print()
        
        self.create_turtle_examples()
        print()
        
        self.create_custom_examples()
        print()
        
        self.create_documentation_examples()
        print()
        
        # Run comprehensive test
        test_results = self.run_comprehensive_test()
        print()
        
        print("=" * 50)
        print("All examples created successfully!")
        print(f"Output directory: {self.output_dir}")
        print(f"Test results: {test_results}")
        
        return test_results


def create_quick_start_example():
    """Create a quick start example for new users."""
    print("Creating quick start example...")
    
    # Simple example
    generator = KolamGenerator()
    pattern = generator.generate('fractal', size=200, complexity=3)
    
    # Visualize
    visualizer = KolamVisualizer()
    visualizer.plot_pattern(pattern, "Quick Start Example", "quick_start.png")
    
    # Analyze
    analyzer = KolamAnalyzer()
    analysis = analyzer.generate_analysis_report(pattern)
    
    print("Quick start example created!")
    print(f"Pattern type: {analysis['classification']['primary_type']}")
    print(f"Complexity: {analysis['classification']['complexity']}")
    print(f"Mathematical properties: {analysis['classification']['mathematical_properties']}")


def create_advanced_example():
    """Create an advanced example showing all features."""
    print("Creating advanced example...")
    
    # Advanced parameters
    params = KolamParameters(
        size=400,
        complexity=5,
        symmetry_type='radial',
        pattern_type='fractal',
        color_scheme='gradient'
    )
    
    generator = KolamGenerator(params)
    pattern = generator.generate()
    
    # Comprehensive analysis
    analyzer = KolamAnalyzer()
    analysis = analyzer.generate_analysis_report(pattern)
    
    # Advanced visualization
    visualizer = KolamVisualizer(style='modern')
    visualizer.plot_pattern(pattern, "Advanced Kolam Example", "advanced_example.png")
    visualizer.plot_3d_pattern(pattern, "3D Advanced Kolam", "advanced_3d.png")
    visualizer.plot_analysis_heatmap(analysis, "advanced_analysis.png")
    
    # Save results
    with open("advanced_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print("Advanced example created!")
    return pattern, analysis


def main():
    """Main function to run all examples."""
    print("Kolam Examples and Test Cases")
    print("=" * 40)
    
    # Create examples instance
    examples = KolamExamples()
    
    # Run all examples
    test_results = examples.run_all_examples()
    
    # Create quick start example
    print("\n" + "=" * 40)
    create_quick_start_example()
    
    # Create advanced example
    print("\n" + "=" * 40)
    create_advanced_example()
    
    print("\n" + "=" * 40)
    print("All examples completed successfully!")
    print("Check the 'kolam_examples' directory for all generated files.")


if __name__ == "__main__":
    main()
