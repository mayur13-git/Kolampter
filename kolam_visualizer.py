"""
Kolam Visualization Tools
========================

This module provides advanced visualization tools for Kolam patterns,
including interactive plots, 3D visualization, and animation capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Polygon
import cv2
from typing import List, Tuple, Dict, Any, Optional
import json
from kolam_analyzer import KolamAnalyzer
from kolam_generator import KolamGenerator, KolamParameters

# Optional imports
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class KolamVisualizer:
    """Advanced visualization tools for Kolam patterns."""
    
    def __init__(self, style: str = 'default'):
        self.style = style
        self.setup_style()
    
    def setup_style(self):
        """Setup matplotlib style."""
        if self.style == 'traditional':
            if HAS_SEABORN:
                plt.style.use('seaborn-v0_8')
            else:
                plt.style.use('default')
            self.colors = ['#8B4513', '#D2691E', '#CD853F', '#DEB887', '#F5DEB3']
        elif self.style == 'modern':
            if HAS_SEABORN:
                plt.style.use('seaborn-v0_8-darkgrid')
            else:
                plt.style.use('default')
            self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        else:
            plt.style.use('default')
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_pattern(self, pattern: np.ndarray, title: str = "Kolam Pattern", 
                    save_path: str = None, show_analysis: bool = False):
        """
        Plot a Kolam pattern with optional analysis overlay.
        
        Args:
            pattern: 2D numpy array representing the Kolam pattern
            title: Title for the plot
            save_path: Path to save the plot
            show_analysis: Whether to show analysis overlay
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot the pattern
        im = ax.imshow(pattern, cmap='viridis', origin='lower')
        
        # Add analysis overlay if requested
        if show_analysis:
            self._add_analysis_overlay(ax, pattern)
        
        # Customize plot
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Pattern Intensity')
        
        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _add_analysis_overlay(self, ax, pattern: np.ndarray):
        """Add analysis overlay to the plot."""
        analyzer = KolamAnalyzer()
        analysis = analyzer.analyze_symmetry(pattern)
        
        # Add symmetry lines
        height, width = pattern.shape
        
        # Radial symmetry center
        if analysis['radial_symmetry']['is_radial']:
            center = analysis['radial_symmetry']['center']
            ax.plot(center[1], center[0], 'r+', markersize=15, markeredgewidth=3)
            
            # Draw radial lines
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                x = center[1] + 50 * np.cos(angle)
                y = center[0] + 50 * np.sin(angle)
                ax.plot([center[1], x], [center[0], y], 'r--', alpha=0.5)
        
        # Bilateral symmetry lines
        if analysis['bilateral_symmetry']['is_bilateral']:
            # Horizontal line
            ax.axhline(y=height//2, color='b', linestyle='--', alpha=0.7)
            # Vertical line
            ax.axvline(x=width//2, color='b', linestyle='--', alpha=0.7)
    
    def plot_comparison(self, patterns: List[np.ndarray], titles: List[str] = None,
                       save_path: str = None):
        """
        Plot multiple Kolam patterns for comparison.
        
        Args:
            patterns: List of 2D numpy arrays
            titles: List of titles for each pattern
            save_path: Path to save the plot
        """
        n_patterns = len(patterns)
        if titles is None:
            titles = [f"Pattern {i+1}" for i in range(n_patterns)]
        
        # Calculate subplot layout
        cols = min(3, n_patterns)
        rows = (n_patterns + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if n_patterns == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (pattern, title) in enumerate(zip(patterns, titles)):
            if i < len(axes):
                im = axes[i].imshow(pattern, cmap='viridis')
                axes[i].set_title(title, fontsize=14)
                axes[i].set_xticks([])
                axes[i].set_yticks([])
        
        # Hide unused subplots
        for i in range(n_patterns, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_3d_pattern(self, pattern: np.ndarray, title: str = "3D Kolam Pattern",
                       save_path: str = None):
        """
        Create a 3D visualization of the Kolam pattern.
        
        Args:
            pattern: 2D numpy array representing the Kolam pattern
            title: Title for the plot
            save_path: Path to save the plot
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create coordinate grids
        height, width = pattern.shape
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        
        # Plot the 3D surface
        surf = ax.plot_surface(X, Y, pattern, cmap='viridis', alpha=0.8)
        
        # Customize the plot
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Pattern Intensity')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_animation(self, patterns: List[np.ndarray], title: str = "Kolam Animation",
                        save_path: str = None, interval: int = 500):
        """
        Create an animation of multiple Kolam patterns.
        
        Args:
            patterns: List of 2D numpy arrays
            title: Title for the animation
            save_path: Path to save the animation
            interval: Animation interval in milliseconds
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Initialize the plot
        im = ax.imshow(patterns[0], cmap='viridis')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        def animate(frame):
            im.set_array(patterns[frame % len(patterns)])
            ax.set_title(f"{title} - Frame {frame + 1}", fontsize=16, fontweight='bold')
            return [im]
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(patterns),
                                     interval=interval, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=2)
        
        plt.show()
        return anim
    
    def plot_analysis_heatmap(self, analysis: Dict[str, Any], save_path: str = None):
        """
        Create a heatmap visualization of analysis results.
        
        Args:
            analysis: Analysis results from KolamAnalyzer
            save_path: Path to save the plot
        """
        # Extract analysis data
        symmetry_data = analysis['symmetry_analysis']
        fractal_data = analysis['fractal_analysis']
        topology_data = analysis['topology_analysis']
        
        # Create data matrix for heatmap
        data = np.array([
            [symmetry_data['radial_symmetry']['score'], 
             symmetry_data['bilateral_symmetry']['max_score'],
             symmetry_data['rotational_symmetry']['best_score']],
            [fractal_data['box_dimension'] / 2,  # Normalize
             fractal_data['self_similarity'],
             fractal_data['recursive_structure']['has_recursive_structure']],
            [topology_data['total_components'] / 10,  # Normalize
             topology_data['total_holes'] / 5,  # Normalize
             topology_data['total_loops'] / 5]  # Normalize
        ])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(3))
        ax.set_xticklabels(['Radial', 'Bilateral', 'Rotational'])
        ax.set_yticks(range(3))
        ax.set_yticklabels(['Symmetry', 'Fractal', 'Topology'])
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Kolam Analysis Heatmap', fontsize=16, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Analysis Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_interactive_3d(self, pattern: np.ndarray, title: str = "Interactive 3D Kolam"):
        """
        Create an interactive 3D plot using Plotly.
        
        Args:
            pattern: 2D numpy array representing the Kolam pattern
            title: Title for the plot
        """
        if not HAS_PLOTLY:
            print("Plotly not available. Install plotly to use interactive 3D plots.")
            return
        
        height, width = pattern.shape
        
        # Create coordinate grids
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(z=pattern, x=X, y=Y, colorscale='Viridis')])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='Pattern Intensity'
            ),
            width=800,
            height=600
        )
        
        fig.show()
    
    def plot_interactive_comparison(self, patterns: List[np.ndarray], 
                                  titles: List[str] = None):
        """
        Create an interactive comparison plot using Plotly.
        
        Args:
            patterns: List of 2D numpy arrays
            titles: List of titles for each pattern
        """
        if not HAS_PLOTLY:
            print("Plotly not available. Install plotly to use interactive comparison plots.")
            return
        
        if titles is None:
            titles = [f"Pattern {i+1}" for i in range(len(patterns))]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=len(patterns),
            subplot_titles=titles,
            specs=[[{"type": "heatmap"} for _ in range(len(patterns))]]
        )
        
        for i, (pattern, title) in enumerate(zip(patterns, titles)):
            fig.add_trace(
                go.Heatmap(z=pattern, colorscale='Viridis', showscale=False),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title="Interactive Kolam Pattern Comparison",
            width=300 * len(patterns),
            height=400
        )
        
        fig.show()
    
    def create_pattern_evolution(self, generator: KolamGenerator, 
                               pattern_type: str, iterations: int = 5,
                               save_path: str = None):
        """
        Create a visualization showing pattern evolution.
        
        Args:
            generator: KolamGenerator instance
            pattern_type: Type of pattern to evolve
            iterations: Number of evolution steps
            save_path: Path to save the animation
        """
        patterns = []
        
        # Generate patterns with increasing complexity
        for i in range(iterations):
            params = KolamParameters(
                size=200,
                complexity=i + 1,
                pattern_type=pattern_type
            )
            generator.params = params
            pattern = generator.generate(pattern_type)
            patterns.append(pattern)
        
        # Create animation
        return self.create_animation(patterns, 
                                   f"{pattern_type.title()} Pattern Evolution",
                                   save_path)
    
    def plot_fractal_analysis(self, pattern: np.ndarray, save_path: str = None):
        """
        Create a detailed fractal analysis visualization.
        
        Args:
            pattern: 2D numpy array representing the Kolam pattern
            save_path: Path to save the plot
        """
        analyzer = KolamAnalyzer()
        fractal_analysis = analyzer.analyze_fractal_properties(pattern)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fractal Analysis of Kolam Pattern', fontsize=16, fontweight='bold')
        
        # Original pattern
        axes[0, 0].imshow(pattern, cmap='viridis')
        axes[0, 0].set_title('Original Pattern')
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        
        # Box-counting analysis
        box_dim = fractal_analysis['box_dimension']
        axes[0, 1].text(0.5, 0.5, f'Box Dimension: {box_dim:.3f}', 
                       ha='center', va='center', fontsize=14, fontweight='bold')
        axes[0, 1].set_title('Fractal Dimension')
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        
        # Self-similarity analysis
        self_sim = fractal_analysis['self_similarity']
        axes[1, 0].text(0.5, 0.5, f'Self-Similarity: {self_sim:.3f}', 
                       ha='center', va='center', fontsize=14, fontweight='bold')
        axes[1, 0].set_title('Self-Similarity Score')
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        
        # Recursive structure
        recursive = fractal_analysis['recursive_structure']
        recursive_text = f"Has Recursive Structure: {recursive['has_recursive_structure']}\n"
        if recursive['grid_periodicity']['has_grid']:
            recursive_text += f"Grid Period: {recursive['grid_periodicity']['period']}\n"
        if recursive['spiral_detection']['has_spiral']:
            recursive_text += f"Spiral Score: {recursive['spiral_detection']['spiral_score']:.3f}"
        
        axes[1, 1].text(0.5, 0.5, recursive_text, 
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Recursive Structure')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_symmetry_visualization(self, pattern: np.ndarray, save_path: str = None):
        """
        Create a detailed symmetry visualization.
        
        Args:
            pattern: 2D numpy array representing the Kolam pattern
            save_path: Path to save the plot
        """
        analyzer = KolamAnalyzer()
        symmetry_analysis = analyzer.analyze_symmetry(pattern)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Symmetry Analysis of Kolam Pattern', fontsize=16, fontweight='bold')
        
        # Original pattern
        axes[0, 0].imshow(pattern, cmap='viridis')
        axes[0, 0].set_title('Original Pattern')
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        
        # Radial symmetry
        radial = symmetry_analysis['radial_symmetry']
        if radial['is_radial']:
            center = radial['center']
            axes[0, 1].imshow(pattern, cmap='viridis')
            axes[0, 1].plot(center[1], center[0], 'r+', markersize=15, markeredgewidth=3)
            axes[0, 1].set_title(f'Radial Symmetry (Score: {radial["score"]:.3f})')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Radial Symmetry', ha='center', va='center')
            axes[0, 1].set_title('Radial Symmetry')
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        
        # Bilateral symmetry
        bilateral = symmetry_analysis['bilateral_symmetry']
        axes[1, 0].imshow(pattern, cmap='viridis')
        if bilateral['is_bilateral']:
            height, width = pattern.shape
            axes[1, 0].axhline(y=height//2, color='b', linestyle='--', alpha=0.7)
            axes[1, 0].axvline(x=width//2, color='b', linestyle='--', alpha=0.7)
            axes[1, 0].set_title(f'Bilateral Symmetry (Score: {bilateral["max_score"]:.3f})')
        else:
            axes[1, 0].set_title('Bilateral Symmetry')
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        
        # Rotational symmetry
        rotational = symmetry_analysis['rotational_symmetry']
        axes[1, 1].text(0.5, 0.5, f'Best Rotational Symmetry: {rotational["best_angle"]}\n'
                                 f'Score: {rotational["best_score"]:.3f}', 
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Rotational Symmetry')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """Example usage of the KolamVisualizer."""
    # Create generator and analyzer
    generator = KolamGenerator()
    analyzer = KolamAnalyzer()
    visualizer = KolamVisualizer(style='modern')
    
    # Generate sample patterns
    patterns = []
    pattern_types = ['fractal', 'spiral', 'geometric', 'traditional']
    
    for pattern_type in pattern_types:
        pattern = generator.generate(pattern_type)
        patterns.append(pattern)
    
    # Basic visualization
    print("Creating basic pattern visualizations...")
    for i, (pattern, pattern_type) in enumerate(zip(patterns, pattern_types)):
        visualizer.plot_pattern(pattern, f"{pattern_type.title()} Kolam Pattern",
                               f'visualization_{pattern_type}.png')
    
    # Comparison plot
    print("Creating comparison plot...")
    visualizer.plot_comparison(patterns, pattern_types, 'pattern_comparison.png')
    
    # 3D visualization
    print("Creating 3D visualization...")
    visualizer.plot_3d_pattern(patterns[0], "3D Fractal Kolam", '3d_fractal_kolam.png')
    
    # Analysis visualizations
    print("Creating analysis visualizations...")
    for i, (pattern, pattern_type) in enumerate(zip(patterns, pattern_types)):
        analysis = analyzer.generate_analysis_report(pattern)
        
        # Analysis heatmap
        visualizer.plot_analysis_heatmap(analysis, f'analysis_heatmap_{pattern_type}.png')
        
        # Fractal analysis
        visualizer.plot_fractal_analysis(pattern, f'fractal_analysis_{pattern_type}.png')
        
        # Symmetry visualization
        visualizer.create_symmetry_visualization(pattern, f'symmetry_analysis_{pattern_type}.png')
    
    # Animation
    print("Creating animation...")
    visualizer.create_animation(patterns, "Kolam Pattern Animation", 'kolam_animation.gif')
    
    # Pattern evolution
    print("Creating pattern evolution...")
    visualizer.create_pattern_evolution(generator, 'fractal', 5, 'fractal_evolution.gif')
    
    # Interactive plots (if plotly is available)
    try:
        print("Creating interactive visualizations...")
        visualizer.plot_interactive_3d(patterns[0], "Interactive 3D Fractal Kolam")
        visualizer.plot_interactive_comparison(patterns, pattern_types)
    except ImportError:
        print("Plotly not available, skipping interactive visualizations")
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()
