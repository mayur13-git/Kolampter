"""
Kolam Design Generator
=====================

This module generates Kolam designs using various mathematical principles
including L-systems, fractals, symmetry, and geometric patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import turtle
import math
from typing import List, Tuple, Dict, Any, Optional
import random
from dataclasses import dataclass
import json


@dataclass
class KolamParameters:
    """Parameters for generating Kolam designs."""
    size: int = 200
    complexity: int = 3
    symmetry_type: str = 'radial'  # 'radial', 'bilateral', 'rotational', 'none'
    pattern_type: str = 'fractal'  # 'fractal', 'spiral', 'grid', 'geometric'
    color_scheme: str = 'monochrome'  # 'monochrome', 'gradient', 'rainbow'
    line_thickness: float = 1.0
    background_color: str = 'white'
    foreground_color: str = 'black'


class LSystem:
    """Lindenmayer System for generating fractal patterns."""
    
    def __init__(self, axiom: str, rules: Dict[str, str]):
        self.axiom = axiom
        self.rules = rules
        self.current_string = axiom
    
    def iterate(self, iterations: int) -> str:
        """Apply production rules for specified iterations."""
        for _ in range(iterations):
            new_string = ""
            for char in self.current_string:
                new_string += self.rules.get(char, char)
            self.current_string = new_string
        return self.current_string
    
    def reset(self):
        """Reset to initial axiom."""
        self.current_string = self.axiom


class KolamGenerator:
    """Main class for generating Kolam designs."""
    
    def __init__(self, parameters: KolamParameters = None):
        self.params = parameters or KolamParameters()
        self.patterns = {
            'fractal': self._generate_fractal_pattern,
            'spiral': self._generate_spiral_pattern,
            'grid': self._generate_grid_pattern,
            'geometric': self._generate_geometric_pattern,
            'traditional': self._generate_traditional_pattern
        }
    
    def generate(self, pattern_type: str = None, **kwargs) -> np.ndarray:
        """
        Generate a Kolam pattern.
        
        Args:
            pattern_type: Type of pattern to generate
            **kwargs: Additional parameters
            
        Returns:
            2D numpy array representing the Kolam pattern
        """
        if pattern_type is None:
            pattern_type = self.params.pattern_type
        
        if pattern_type not in self.patterns:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        # Update parameters with kwargs
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
        
        # Generate the base pattern
        pattern = self.patterns[pattern_type]()
        
        # Apply symmetry if specified
        if self.params.symmetry_type != 'none':
            pattern = self._apply_symmetry(pattern)
        
        # Apply color scheme
        pattern = self._apply_color_scheme(pattern)
        
        return pattern
    
    def _generate_fractal_pattern(self) -> np.ndarray:
        """Generate a fractal-based Kolam pattern using L-systems."""
        # Define L-system for fractal kolam
        lsystem = LSystem(
            axiom="F+F+F+F",
            rules={
                "F": "F+F-F-F+F",
                "+": "+",
                "-": "-"
            }
        )
        
        # Generate the string
        instructions = lsystem.iterate(self.params.complexity)
        
        # Convert to pattern
        pattern = self._lstring_to_pattern(instructions)
        
        return pattern
    
    def _generate_spiral_pattern(self) -> np.ndarray:
        """Generate a spiral-based Kolam pattern."""
        pattern = np.zeros((self.params.size, self.params.size))
        center = (self.params.size // 2, self.params.size // 2)
        
        # Create spiral parameters
        num_spirals = 4 + self.params.complexity
        max_radius = self.params.size // 2 - 10
        
        for spiral_id in range(num_spirals):
            # Spiral parameters
            start_angle = spiral_id * 2 * np.pi / num_spirals
            spiral_tightness = 0.1 + (spiral_id % 3) * 0.05
            
            # Generate spiral points
            points = self._generate_spiral_points(
                center, start_angle, spiral_tightness, max_radius
            )
            
            # Draw spiral
            for i in range(len(points) - 1):
                self._draw_line(pattern, points[i], points[i + 1])
        
        return pattern
    
    def _generate_grid_pattern(self) -> np.ndarray:
        """Generate a grid-based Kolam pattern."""
        pattern = np.zeros((self.params.size, self.params.size))
        
        # Grid parameters
        grid_size = 20 + self.params.complexity * 5
        num_rows = self.params.size // grid_size
        num_cols = self.params.size // grid_size
        
        # Generate grid motifs
        for row in range(num_rows):
            for col in range(num_cols):
                x = col * grid_size
                y = row * grid_size
                
                # Generate motif in this grid cell
                motif = self._generate_grid_motif(grid_size)
                
                # Place motif in pattern
                end_y = min(y + grid_size, self.params.size)
                end_x = min(x + grid_size, self.params.size)
                pattern[y:end_y, x:end_x] = motif[:end_y-y, :end_x-x]
        
        return pattern
    
    def _generate_geometric_pattern(self) -> np.ndarray:
        """Generate a geometric Kolam pattern."""
        pattern = np.zeros((self.params.size, self.params.size))
        center = (self.params.size // 2, self.params.size // 2)
        
        # Generate concentric geometric shapes
        num_shapes = 3 + self.params.complexity
        max_radius = self.params.size // 2 - 10
        
        for i in range(num_shapes):
            radius = (i + 1) * max_radius // num_shapes
            sides = 6 + (i % 3) * 2  # 6, 8, or 10 sides
            
            # Generate polygon
            polygon_points = self._generate_polygon(center, radius, sides)
            
            # Draw polygon
            for j in range(len(polygon_points)):
                start = polygon_points[j]
                end = polygon_points[(j + 1) % len(polygon_points)]
                self._draw_line(pattern, start, end)
        
        return pattern
    
    def _generate_traditional_pattern(self) -> np.ndarray:
        """Generate a traditional Kolam pattern."""
        pattern = np.zeros((self.params.size, self.params.size))
        center = (self.params.size // 2, self.params.size // 2)
        
        # Traditional kolam elements
        # Central lotus
        self._draw_lotus(pattern, center, 30)
        
        # Surrounding petals
        num_petals = 8
        for i in range(num_petals):
            angle = i * 2 * np.pi / num_petals
            petal_center = (
                center[0] + int(50 * np.cos(angle)),
                center[1] + int(50 * np.sin(angle))
            )
            self._draw_petal(pattern, petal_center, 15, angle)
        
        # Outer border
        self._draw_circle(pattern, center, 80)
        
        return pattern
    
    def _lstring_to_pattern(self, instructions: str) -> np.ndarray:
        """Convert L-system string to pattern."""
        pattern = np.zeros((self.params.size, self.params.size))
        
        # Turtle graphics simulation
        x, y = self.params.size // 2, self.params.size // 2
        angle = 0
        stack = []
        
        step_size = 5
        angle_step = 90
        
        for command in instructions:
            if command == 'F':
                # Move forward
                new_x = x + step_size * np.cos(np.radians(angle))
                new_y = y + step_size * np.sin(np.radians(angle))
                
                # Draw line
                self._draw_line(pattern, (int(y), int(x)), (int(new_y), int(new_x)))
                
                x, y = new_x, new_y
                
            elif command == '+':
                angle += angle_step
            elif command == '-':
                angle -= angle_step
            elif command == '[':
                stack.append((x, y, angle))
            elif command == ']':
                if stack:
                    x, y, angle = stack.pop()
        
        return pattern
    
    def _generate_spiral_points(self, center: Tuple[int, int], start_angle: float, 
                               tightness: float, max_radius: int) -> List[Tuple[int, int]]:
        """Generate points for a spiral."""
        points = []
        angle = start_angle
        radius = 5
        
        while radius < max_radius:
            x = center[1] + radius * np.cos(angle)
            y = center[0] + radius * np.sin(angle)
            
            if 0 <= x < self.params.size and 0 <= y < self.params.size:
                points.append((int(y), int(x)))
            
            angle += tightness
            radius += 0.5
        
        return points
    
    def _generate_grid_motif(self, size: int) -> np.ndarray:
        """Generate a motif for grid pattern."""
        motif = np.zeros((size, size))
        center = (size // 2, size // 2)
        
        # Random motif type
        motif_type = random.choice(['dot', 'cross', 'diamond', 'star'])
        
        if motif_type == 'dot':
            self._draw_circle(motif, center, size // 4)
        elif motif_type == 'cross':
            self._draw_cross(motif, center, size // 3)
        elif motif_type == 'diamond':
            self._draw_diamond(motif, center, size // 3)
        elif motif_type == 'star':
            self._draw_star(motif, center, size // 3)
        
        return motif
    
    def _generate_polygon(self, center: Tuple[int, int], radius: int, sides: int) -> List[Tuple[int, int]]:
        """Generate points for a regular polygon."""
        points = []
        for i in range(sides):
            angle = i * 2 * np.pi / sides
            x = center[1] + radius * np.cos(angle)
            y = center[0] + radius * np.sin(angle)
            points.append((int(y), int(x)))
        return points
    
    def _draw_line(self, pattern: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]):
        """Draw a line between two points."""
        y1, x1 = start
        y2, x2 = end
        
        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            if 0 <= x1 < pattern.shape[1] and 0 <= y1 < pattern.shape[0]:
                pattern[y1, x1] = 1
            
            if x1 == x2 and y1 == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
    
    def _draw_circle(self, pattern: np.ndarray, center: Tuple[int, int], radius: int):
        """Draw a circle."""
        y_center, x_center = center
        
        for angle in np.linspace(0, 2 * np.pi, 100):
            x = int(x_center + radius * np.cos(angle))
            y = int(y_center + radius * np.sin(angle))
            
            if 0 <= x < pattern.shape[1] and 0 <= y < pattern.shape[0]:
                pattern[y, x] = 1
    
    def _draw_cross(self, pattern: np.ndarray, center: Tuple[int, int], size: int):
        """Draw a cross."""
        y_center, x_center = center
        half_size = size // 2
        
        # Vertical line
        for y in range(y_center - half_size, y_center + half_size + 1):
            if 0 <= y < pattern.shape[0]:
                pattern[y, x_center] = 1
        
        # Horizontal line
        for x in range(x_center - half_size, x_center + half_size + 1):
            if 0 <= x < pattern.shape[1]:
                pattern[y_center, x] = 1
    
    def _draw_diamond(self, pattern: np.ndarray, center: Tuple[int, int], size: int):
        """Draw a diamond."""
        y_center, x_center = center
        half_size = size // 2
        
        # Diamond vertices
        vertices = [
            (y_center - half_size, x_center),  # top
            (y_center, x_center + half_size),  # right
            (y_center + half_size, x_center),  # bottom
            (y_center, x_center - half_size)   # left
        ]
        
        # Draw diamond edges
        for i in range(len(vertices)):
            start = vertices[i]
            end = vertices[(i + 1) % len(vertices)]
            self._draw_line(pattern, start, end)
    
    def _draw_star(self, pattern: np.ndarray, center: Tuple[int, int], size: int):
        """Draw a star."""
        y_center, x_center = center
        outer_radius = size // 2
        inner_radius = size // 4
        
        # Generate star points
        points = []
        for i in range(10):  # 5-pointed star
            angle = i * np.pi / 5
            if i % 2 == 0:
                radius = outer_radius
            else:
                radius = inner_radius
            
            x = x_center + radius * np.cos(angle)
            y = y_center + radius * np.sin(angle)
            points.append((int(y), int(x)))
        
        # Draw star edges
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            self._draw_line(pattern, start, end)
    
    def _draw_lotus(self, pattern: np.ndarray, center: Tuple[int, int], radius: int):
        """Draw a lotus pattern."""
        y_center, x_center = center
        
        # Draw lotus petals
        num_petals = 8
        for i in range(num_petals):
            angle = i * 2 * np.pi / num_petals
            
            # Petal points
            petal_points = []
            for t in np.linspace(0, 1, 20):
                petal_radius = radius * t * (1 - t) * 4  # Petal shape
                x = x_center + petal_radius * np.cos(angle)
                y = y_center + petal_radius * np.sin(angle)
                petal_points.append((int(y), int(x)))
            
            # Draw petal
            for j in range(len(petal_points) - 1):
                self._draw_line(pattern, petal_points[j], petal_points[j + 1])
    
    def _draw_petal(self, pattern: np.ndarray, center: Tuple[int, int], size: int, angle: float):
        """Draw a petal."""
        y_center, x_center = center
        
        # Petal shape
        petal_points = []
        for t in np.linspace(0, 1, 15):
            petal_radius = size * t * (1 - t) * 4
            x = x_center + petal_radius * np.cos(angle)
            y = y_center + petal_radius * np.sin(angle)
            petal_points.append((int(y), int(x)))
        
        # Draw petal
        for j in range(len(petal_points) - 1):
            self._draw_line(pattern, petal_points[j], petal_points[j + 1])
    
    def _apply_symmetry(self, pattern: np.ndarray) -> np.ndarray:
        """Apply symmetry to the pattern."""
        if self.params.symmetry_type == 'radial':
            return self._apply_radial_symmetry(pattern)
        elif self.params.symmetry_type == 'bilateral':
            return self._apply_bilateral_symmetry(pattern)
        elif self.params.symmetry_type == 'rotational':
            return self._apply_rotational_symmetry(pattern)
        else:
            return pattern
    
    def _apply_radial_symmetry(self, pattern: np.ndarray) -> np.ndarray:
        """Apply radial symmetry."""
        # Take one quadrant and mirror it
        height, width = pattern.shape
        h_mid, w_mid = height // 2, width // 2
        
        # Use top-left quadrant as base
        base_quadrant = pattern[:h_mid, :w_mid]
        
        # Mirror to other quadrants
        pattern[:h_mid, w_mid:] = np.fliplr(base_quadrant)
        pattern[h_mid:, :w_mid] = np.flipud(base_quadrant)
        pattern[h_mid:, w_mid:] = np.flipud(np.fliplr(base_quadrant))
        
        return pattern
    
    def _apply_bilateral_symmetry(self, pattern: np.ndarray) -> np.ndarray:
        """Apply bilateral symmetry."""
        height, width = pattern.shape
        
        # Horizontal symmetry
        if height % 2 == 0:
            top_half = pattern[:height//2, :]
            pattern[height//2:, :] = np.flipud(top_half)
        
        # Vertical symmetry
        if width % 2 == 0:
            left_half = pattern[:, :width//2]
            pattern[:, width//2:] = np.fliplr(left_half)
        
        return pattern
    
    def _apply_rotational_symmetry(self, pattern: np.ndarray) -> np.ndarray:
        """Apply rotational symmetry."""
        # Create 4-fold rotational symmetry
        height, width = pattern.shape
        h_mid, w_mid = height // 2, width // 2
        
        # Use top-right quadrant as base
        base_quadrant = pattern[:h_mid, w_mid:]
        
        # Rotate and place in other quadrants
        pattern[:h_mid, :w_mid] = np.rot90(base_quadrant, 1)
        pattern[h_mid:, w_mid:] = np.rot90(base_quadrant, 2)
        pattern[h_mid:, :w_mid] = np.rot90(base_quadrant, 3)
        
        return pattern
    
    def _apply_color_scheme(self, pattern: np.ndarray) -> np.ndarray:
        """Apply color scheme to the pattern."""
        if self.params.color_scheme == 'monochrome':
            return pattern
        elif self.params.color_scheme == 'gradient':
            return self._apply_gradient(pattern)
        elif self.params.color_scheme == 'rainbow':
            return self._apply_rainbow(pattern)
        else:
            return pattern
    
    def _apply_gradient(self, pattern: np.ndarray) -> np.ndarray:
        """Apply gradient color scheme."""
        # Create distance-based gradient
        height, width = pattern.shape
        center = (height // 2, width // 2)
        
        y, x = np.ogrid[:height, :width]
        distances = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        max_distance = np.max(distances)
        
        # Normalize distances
        normalized_distances = distances / max_distance
        
        # Apply gradient
        gradient_pattern = pattern * normalized_distances
        
        return gradient_pattern
    
    def _apply_rainbow(self, pattern: np.ndarray) -> np.ndarray:
        """Apply rainbow color scheme."""
        # Create angle-based rainbow
        height, width = pattern.shape
        center = (height // 2, width // 2)
        
        y, x = np.ogrid[:height, :width]
        angles = np.arctan2(y - center[0], x - center[1])
        
        # Normalize angles to 0-1
        normalized_angles = (angles + np.pi) / (2 * np.pi)
        
        # Apply rainbow
        rainbow_pattern = pattern * normalized_angles
        
        return rainbow_pattern
    
    def save_pattern(self, pattern: np.ndarray, filename: str):
        """Save pattern to file."""
        plt.figure(figsize=(10, 10))
        plt.imshow(pattern, cmap='viridis')
        plt.axis('off')
        plt.title('Generated Kolam Pattern')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_batch(self, num_patterns: int, pattern_types: List[str] = None) -> List[np.ndarray]:
        """Generate multiple patterns."""
        if pattern_types is None:
            pattern_types = list(self.patterns.keys())
        
        patterns = []
        for i in range(num_patterns):
            pattern_type = random.choice(pattern_types)
            pattern = self.generate(pattern_type)
            patterns.append(pattern)
        
        return patterns


class TurtleKolamGenerator:
    """Kolam generator using Turtle graphics for interactive visualization."""
    
    def __init__(self, size: int = 400):
        self.size = size
        self.turtle = turtle.Turtle()
        self.screen = turtle.Screen()
        self.setup_turtle()
    
    def setup_turtle(self):
        """Setup turtle graphics."""
        self.screen.setup(self.size, self.size)
        self.screen.bgcolor('white')
        self.turtle.speed(0)
        self.turtle.penup()
        self.turtle.goto(0, 0)
        self.turtle.pendown()
    
    def draw_fractal_kolam(self, iterations: int = 3):
        """Draw a fractal kolam using L-systems."""
        lsystem = LSystem(
            axiom="F+F+F+F",
            rules={"F": "F+F-F-F+F", "+": "+", "-": "-"}
        )
        
        instructions = lsystem.iterate(iterations)
        
        # Draw the pattern
        step_size = 5
        angle_step = 90
        
        for command in instructions:
            if command == 'F':
                self.turtle.forward(step_size)
            elif command == '+':
                self.turtle.right(angle_step)
            elif command == '-':
                self.turtle.left(angle_step)
    
    def draw_spiral_kolam(self, num_spirals: int = 4):
        """Draw a spiral kolam."""
        for i in range(num_spirals):
            self.turtle.penup()
            self.turtle.goto(0, 0)
            self.turtle.pendown()
            
            # Draw spiral
            for j in range(100):
                self.turtle.forward(j * 0.1)
                self.turtle.right(90 + i * 10)
    
    def draw_geometric_kolam(self, num_sides: int = 8):
        """Draw a geometric kolam."""
        for i in range(5):
            self.turtle.penup()
            self.turtle.goto(0, 0)
            self.turtle.pendown()
            
            # Draw polygon
            for j in range(num_sides):
                self.turtle.forward(50 + i * 10)
                self.turtle.right(360 / num_sides)
    
    def save_screen(self, filename: str):
        """Save the turtle screen."""
        self.screen.getcanvas().postscript(file=filename)
    
    def close(self):
        """Close the turtle screen."""
        self.screen.bye()


def main():
    """Example usage of the KolamGenerator."""
    # Create generator with custom parameters
    params = KolamParameters(
        size=300,
        complexity=4,
        symmetry_type='radial',
        pattern_type='fractal',
        color_scheme='gradient'
    )
    
    generator = KolamGenerator(params)
    
    # Generate different types of patterns
    pattern_types = ['fractal', 'spiral', 'grid', 'geometric', 'traditional']
    
    for i, pattern_type in enumerate(pattern_types):
        print(f"Generating {pattern_type} pattern...")
        pattern = generator.generate(pattern_type)
        generator.save_pattern(pattern, f'kolam_{pattern_type}_{i+1}.png')
    
    # Generate batch of random patterns
    print("Generating batch of random patterns...")
    random_patterns = generator.generate_batch(5)
    
    for i, pattern in enumerate(random_patterns):
        generator.save_pattern(pattern, f'kolam_random_{i+1}.png')
    
    print("Pattern generation complete!")
    
    # Interactive turtle example
    print("Starting interactive turtle demonstration...")
    turtle_gen = TurtleKolamGenerator()
    
    # Draw different patterns
    turtle_gen.draw_fractal_kolam(3)
    turtle_gen.save_screen('turtle_fractal_kolam.eps')
    
    turtle_gen.screen.clear()
    turtle_gen.draw_spiral_kolam(4)
    turtle_gen.save_screen('turtle_spiral_kolam.eps')
    
    turtle_gen.screen.clear()
    turtle_gen.draw_geometric_kolam(8)
    turtle_gen.save_screen('turtle_geometric_kolam.eps')
    
    turtle_gen.close()


if __name__ == "__main__":
    main()
