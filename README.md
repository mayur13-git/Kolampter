# Kolam Design Analysis and Generation System

A comprehensive Python system for analyzing and generating traditional South Indian Kolam designs using mathematical principles including fractals, symmetry, topology, and geometric patterns.

## Overview

Kolam is a traditional South Indian art form characterized by intricate geometric patterns drawn using rice flour or chalk powder. These designs often exhibit mathematical properties such as symmetry, fractals, and tessellations. This system provides tools to:

- **Analyze** existing Kolam patterns to identify mathematical principles
- **Generate** new Kolam designs using various algorithms
- **Visualize** patterns in 2D, 3D, and interactive formats
- **Study** the mathematical foundations of traditional art

## Features

### 🔍 Pattern Analysis
- **Symmetry Analysis**: Radial, bilateral, and rotational symmetry detection
- **Fractal Analysis**: Box-counting dimension and self-similarity measurement
- **Topology Analysis**: Connected components, holes, and loops detection
- **Pattern Classification**: Automatic categorization based on mathematical properties

### 🎨 Pattern Generation
- **L-System Fractals**: Lindenmayer systems for recursive patterns
- **Spiral Patterns**: Mathematical spiral generation
- **Grid Patterns**: Tessellated designs with repeating motifs
- **Geometric Patterns**: Regular polygons and geometric shapes
- **Traditional Patterns**: Classic Kolam designs

### 📊 Visualization Tools
- **2D Visualization**: High-quality pattern rendering
- **3D Visualization**: Surface plots and interactive 3D models
- **Animation**: Pattern evolution and transformation sequences
- **Interactive Plots**: Plotly-based interactive visualizations
- **Analysis Overlays**: Symmetry lines and mathematical annotations

### 🐢 Interactive Graphics
- **Turtle Graphics**: Interactive pattern drawing
- **Real-time Generation**: Live pattern creation and modification
- **Educational Tools**: Step-by-step pattern construction

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Kolampter
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Pattern Generation

```python
from kolam_generator import KolamGenerator, KolamParameters

# Create a generator
generator = KolamGenerator()

# Generate a fractal pattern
pattern = generator.generate('fractal', size=300, complexity=4)

# Save the pattern
generator.save_pattern(pattern, 'my_kolam.png')
```

### Pattern Analysis

```python
from kolam_analyzer import KolamAnalyzer

# Create an analyzer
analyzer = KolamAnalyzer()

# Analyze a pattern
analysis = analyzer.generate_analysis_report(pattern)

# Print results
print(f"Pattern type: {analysis['classification']['primary_type']}")
print(f"Complexity: {analysis['classification']['complexity']}")
print(f"Mathematical properties: {analysis['classification']['mathematical_properties']}")
```

### Visualization

```python
from kolam_visualizer import KolamVisualizer

# Create a visualizer
visualizer = KolamVisualizer()

# Plot the pattern
visualizer.plot_pattern(pattern, "My Kolam Pattern")

# Create 3D visualization
visualizer.plot_3d_pattern(pattern, "3D Kolam")

# Generate analysis heatmap
visualizer.plot_analysis_heatmap(analysis, "analysis.png")
```

## Usage Examples

### 1. Generate Different Pattern Types

```python
from kolam_generator import KolamGenerator

generator = KolamGenerator()

# Fractal pattern
fractal_pattern = generator.generate('fractal', complexity=4)

# Spiral pattern
spiral_pattern = generator.generate('spiral', complexity=3)

# Grid pattern
grid_pattern = generator.generate('grid', complexity=2)

# Geometric pattern
geometric_pattern = generator.generate('geometric', complexity=3)

# Traditional pattern
traditional_pattern = generator.generate('traditional', complexity=3)
```

### 2. Custom Parameters

```python
from kolam_generator import KolamParameters

# Custom parameters
params = KolamParameters(
    size=400,
    complexity=5,
    symmetry_type='radial',
    pattern_type='fractal',
    color_scheme='gradient'
)

generator = KolamGenerator(params)
pattern = generator.generate()
```

### 3. Comprehensive Analysis

```python
from kolam_analyzer import KolamAnalyzer

analyzer = KolamAnalyzer()
analysis = analyzer.generate_analysis_report(pattern)

# Access specific analysis results
symmetry = analysis['symmetry_analysis']
fractal = analysis['fractal_analysis']
topology = analysis['topology_analysis']

print(f"Radial symmetry score: {symmetry['radial_symmetry']['score']}")
print(f"Fractal dimension: {fractal['box_dimension']}")
print(f"Number of components: {topology['total_components']}")
```

### 4. Interactive Turtle Graphics

```python
from kolam_generator import TurtleKolamGenerator

# Create interactive turtle graphics
turtle_gen = TurtleKolamGenerator(400)

# Draw different patterns
turtle_gen.draw_fractal_kolam(3)
turtle_gen.draw_spiral_kolam(4)
turtle_gen.draw_geometric_kolam(8)

# Save the result
turtle_gen.save_screen('turtle_kolam.eps')
turtle_gen.close()
```

### 5. Batch Generation

```python
# Generate multiple patterns
patterns = generator.generate_batch(10, ['fractal', 'spiral', 'geometric'])

# Visualize comparison
visualizer.plot_comparison(patterns, save_path='comparison.png')
```

## Mathematical Principles

### Symmetry Analysis
- **Radial Symmetry**: Patterns symmetric around a central point
- **Bilateral Symmetry**: Mirror symmetry across axes
- **Rotational Symmetry**: Patterns that repeat under rotation

### Fractal Properties
- **Box-Counting Dimension**: Measures fractal complexity
- **Self-Similarity**: Patterns that repeat at different scales
- **Recursive Structure**: Hierarchical pattern organization

### Topology
- **Connected Components**: Number of separate pattern elements
- **Holes**: Enclosed regions within patterns
- **Loops**: Closed curves in the pattern

## File Structure

```
Kolampter/
├── kolam_analyzer.py      # Pattern analysis tools
├── kolam_generator.py     # Pattern generation algorithms
├── kolam_visualizer.py    # Visualization tools
├── examples.py           # Example patterns and test cases
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## API Reference

### KolamGenerator

Main class for generating Kolam patterns.

**Methods:**
- `generate(pattern_type, **kwargs)`: Generate a pattern
- `generate_batch(num_patterns, pattern_types)`: Generate multiple patterns
- `save_pattern(pattern, filename)`: Save pattern to file

### KolamAnalyzer

Main class for analyzing Kolam patterns.

**Methods:**
- `analyze_symmetry(pattern)`: Analyze symmetry properties
- `analyze_fractal_properties(pattern)`: Analyze fractal properties
- `analyze_topology(pattern)`: Analyze topological properties
- `generate_analysis_report(pattern)`: Generate comprehensive analysis

### KolamVisualizer

Main class for visualizing Kolam patterns.

**Methods:**
- `plot_pattern(pattern, title, save_path)`: Plot 2D pattern
- `plot_3d_pattern(pattern, title, save_path)`: Plot 3D pattern
- `plot_comparison(patterns, titles, save_path)`: Compare multiple patterns
- `create_animation(patterns, title, save_path)`: Create animation
- `plot_analysis_heatmap(analysis, save_path)`: Plot analysis results

## Examples and Test Cases

Run the examples to see the system in action:

```bash
python examples.py
```

This will create:
- Basic pattern examples
- Comparison visualizations
- Analysis examples
- Animation sequences
- Turtle graphics examples
- Custom pattern examples

## Advanced Usage

### Custom L-Systems

```python
from kolam_generator import LSystem

# Define custom L-system
lsystem = LSystem(
    axiom="F+F+F+F",
    rules={
        "F": "F+F-F-F+F",
        "+": "+",
        "-": "-"
    }
)

# Generate pattern
instructions = lsystem.iterate(4)
pattern = generator._lstring_to_pattern(instructions)
```

### Custom Visualization Styles

```python
# Traditional style
visualizer = KolamVisualizer(style='traditional')

# Modern style
visualizer = KolamVisualizer(style='modern')

# Default style
visualizer = KolamVisualizer(style='default')
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- New pattern generation algorithms
- Additional analysis methods
- Visualization improvements
- Documentation enhancements
- Bug fixes

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Traditional Kolam artists and their mathematical insights
- Mathematical research on fractals and symmetry
- The Python scientific computing community
- Open source visualization libraries

## References

1. "The Significance of Kolam in Tamil Culture" - Sahapedia
2. "Mastering the Art of Drawing a Padi Kolam" - Desert Divers
3. "Margazhi Kolam" - Sai Tech Info
4. "L-System for Single Knot Kolam Pattern Generation" - Imaginary.org
5. "KolamGenerator" - GitHub repository by trishar

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on the repository.

---

*This system combines traditional art with modern computational methods to explore the mathematical beauty of Kolam designs.*
