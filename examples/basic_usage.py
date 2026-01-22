"""
Basic MVEE-Opt Usage Example
============================

Demonstrates core functionality of the MVEE module:
- Computing minimum volume enclosing ellipsoid
- Accessing geometric parameters
- Visualizing results with matplotlib

Author: TonyLuna
Date: January 21, 2026
License: MIT

Requirements:
    uv pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Import from parent directory
import sys
sys.path.insert(0, '..')
from mvee import MVEE


def main():
    # Generate sample data: elongated cluster with some outliers
    np.random.seed(42)
    
    # Main cluster
    n_points = 150
    angle = np.radians(30)  # 30 degree rotation
    cluster = np.random.randn(n_points, 2) @ [[3, 0], [0, 1]]  # Elongated
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    cluster = cluster @ rotation + [2, 1]  # Rotate and shift
    
    # Add a few outliers
    outliers = np.array([[8, 5], [-4, -2], [6, -3]])
    points = np.vstack([cluster, outliers])
    
    # Compute MVEE
    solver = MVEE(tol=0.01)
    result = solver.fit(points)
    
    # Extract parameters
    center = result.ellipse.center
    radii = result.ellipse.radii
    rot_matrix = result.ellipse.rotation
    ellipse_angle = np.degrees(np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0]))
    
    # Print results
    print("MVEE-Opt Results")
    print("=" * 40)
    print(f"Center: ({center[0]:.3f}, {center[1]:.3f})")
    print(f"Semi-axes: a={radii[0]:.3f}, b={radii[1]:.3f}")
    print(f"Rotation: {ellipse_angle:.1f}°")
    print(f"Area: {np.pi * radii[0] * radii[1]:.2f}")
    print(f"Eccentricity: {np.sqrt(1 - (radii[1]/radii[0])**2):.3f}")
    print(f"Core set size: {len(result.core_set_indices)} points")
    print(f"Converged: {result.converged} in {result.iterations} iterations")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all points
    ax.scatter(points[:, 0], points[:, 1], c='steelblue', s=30, 
               alpha=0.6, label='Points')
    
    # Highlight core set (boundary points)
    core_points = points[result.core_set_indices]
    ax.scatter(core_points[:, 0], core_points[:, 1], c='red', s=100, 
               marker='*', label=f'Core set ({len(core_points)} pts)', zorder=5)
    
    # Draw ellipse
    ellipse = Ellipse(xy=center, width=2*radii[0], height=2*radii[1],
                      angle=ellipse_angle, fill=False, 
                      edgecolor='green', linewidth=2.5, label='MVEE')
    ax.add_patch(ellipse)
    
    # Mark center
    ax.scatter(*center, c='green', s=150, marker='+', linewidths=3, 
               label='Center', zorder=6)
    
    # Draw semi-axes
    for i, (length, color) in enumerate(zip(radii, ['darkgreen', 'lightgreen'])):
        direction = rot_matrix[:, i] * length
        ax.annotate('', xy=center + direction, xytext=center,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_title('Minimum Volume Enclosing Ellipsoid (MVEE)', fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig('mvee_example.png', dpi=150)
    plt.show()
    
    print("\nVisualization saved to 'mvee_example.png'")


if __name__ == "__main__":
    main()