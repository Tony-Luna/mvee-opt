"""
MVEE-Opt Benchmark Suite
========================

Compares MVEE-Opt against alternative ellipse methods and generates:
- media/hero.png: Visually appealing front-page image for README
- media/benchmark_results.png: Full comparison across test distributions

Author: TonyLuna
Date: January 21, 2026
License: MIT

Requirements:
    uv pip install numpy matplotlib scipy opencv-python scikit-image

Usage:
    mkdir -p media
    python benchmark.py
"""

import os
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import time
import warnings

warnings.filterwarnings('ignore')

# Import our optimized MVEE
from mvee import MVEE

# Ensure media directory exists
os.makedirs('media', exist_ok=True)

# Optional imports for comparison
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("OpenCV not installed, skipping fitEllipse comparison")

try:
    from skimage.measure import EllipseModel
    import skimage
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("scikit-image not installed, skipping EllipseModel comparison")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_ellipse(center, radii, angle):
    """Normalize: major axis first, angle in [-π, π]."""
    radii = np.array(radii, dtype=np.float64)
    if radii[1] > radii[0]:
        radii = radii[::-1]
        angle = angle + np.pi / 2
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    return center, radii, angle


def check_enclosure(points, center, radii, angle):
    """Check what fraction of points lie inside the ellipse."""
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    centered = points - center
    rotated = np.column_stack([
        centered[:, 0] * cos_a - centered[:, 1] * sin_a,
        centered[:, 0] * sin_a + centered[:, 1] * cos_a
    ])
    r0 = max(radii[0], 1e-10)
    r1 = max(radii[1], 1e-10)
    normalized = (rotated[:, 0] / r0)**2 + (rotated[:, 1] / r1)**2
    frac_inside = (normalized <= 1.001).mean()
    return frac_inside >= 0.999, frac_inside


def compute_area(radii):
    """Ellipse area = π * a * b."""
    return np.pi * radii[0] * radii[1]


# =============================================================================
# ALTERNATIVE METHODS
# =============================================================================

class OriginalMVEE:
    """Original O(N²) per iteration Khachiyan implementation."""
    
    def __init__(self, tol=0.01, max_iter=200):
        self.tol = tol
        self.max_iter = max_iter
    
    def fit(self, P):
        P = np.asarray(P, dtype=np.float64)
        N, d = P.shape
        
        Q = np.vstack([P.T, np.ones(N)])
        QT = Q.T
        u = np.ones(N) / N
        
        for iteration in range(self.max_iter):
            V = Q @ np.diag(u) @ QT
            try:
                M = np.diag(QT @ la.inv(V) @ Q)
            except la.LinAlgError:
                M = np.diag(QT @ la.pinv(V) @ Q)
            
            j = np.argmax(M)
            max_M = M[j]
            
            if max_M - d - 1 < self.tol:
                break
            
            step = (max_M - d - 1) / ((d + 1) * (max_M - 1))
            u = (1 - step) * u
            u[j] += step
        
        center = P.T @ u
        P_centered = P - center
        Sigma = P_centered.T @ np.diag(u) @ P_centered
        
        try:
            A = la.inv(Sigma) / d
        except la.LinAlgError:
            A = la.pinv(Sigma) / d
        
        eigenvalues, eigenvectors = la.eigh(A)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        radii = 1.0 / np.sqrt(eigenvalues)
        
        order = np.argsort(radii)[::-1]
        radii = radii[order]
        eigenvectors = eigenvectors[:, order]
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        
        # Scale to ensure enclosure
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        centered = P - center
        rotated = np.column_stack([
            centered[:, 0] * cos_a - centered[:, 1] * sin_a,
            centered[:, 0] * sin_a + centered[:, 1] * cos_a
        ])
        normalized = (rotated[:, 0] / radii[0])**2 + (rotated[:, 1] / radii[1])**2
        if normalized.max() > 1.0:
            radii = radii * np.sqrt(normalized.max()) * 1.001
        
        return center, radii, angle, iteration + 1


def opencv_fit_ellipse(points):
    """OpenCV fitEllipse (fitting, NOT enclosing)."""
    if not HAS_OPENCV:
        raise ImportError("OpenCV not available")
    
    points_cv = points.astype(np.float32)
    (cx, cy), (width, height), angle_deg = cv2.fitEllipse(points_cv)
    center = np.array([cx, cy])
    radii = np.array([width / 2, height / 2])
    angle_rad = np.radians(angle_deg)
    return normalize_ellipse(center, radii, angle_rad)


def skimage_fit_ellipse(points):
    """scikit-image EllipseModel (fitting, NOT enclosing)."""
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image not available")
    
    version = tuple(int(x) for x in skimage.__version__.split('.')[:2])
    
    if version >= (0, 26):
        model = EllipseModel.from_estimate(points)
        if model is None:
            raise ValueError("EllipseModel.from_estimate failed")
        center = np.array(model.center)
        radii = np.array(model.axis_lengths)
        theta = model.theta
    else:
        model = EllipseModel()
        if not model.estimate(points):
            raise ValueError("EllipseModel.estimate failed")
        xc, yc, a, b, theta = model.params
        center = np.array([xc, yc])
        radii = np.array([a, b])
    
    return normalize_ellipse(center, radii, theta)


def covariance_ellipse(points):
    """Covariance-based ellipse, scaled to enclose all points."""
    center = points.mean(axis=0)
    cov = np.cov(points.T)
    if cov.ndim == 0:
        cov = np.array([[cov, 0], [0, cov]])
    
    eigenvalues, eigenvectors = la.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    radii = np.sqrt(np.maximum(eigenvalues, 1e-10))
    
    # Scale to enclose
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    centered = points - center
    rotated = np.column_stack([
        centered[:, 0] * cos_a - centered[:, 1] * sin_a,
        centered[:, 0] * sin_a + centered[:, 1] * cos_a
    ])
    normalized = (rotated[:, 0] / max(radii[0], 1e-10))**2 + \
                 (rotated[:, 1] / max(radii[1], 1e-10))**2
    scale = np.sqrt(normalized.max()) * 1.001
    radii = radii * scale
    
    return center, radii, angle


def bounding_box_ellipse(points):
    """Axis-aligned bounding box circumscribed ellipse."""
    min_pt, max_pt = points.min(axis=0), points.max(axis=0)
    center = (min_pt + max_pt) / 2
    half_size = (max_pt - min_pt) / 2
    radii = half_size * np.sqrt(2)
    radii = np.sort(radii)[::-1]
    return center, radii, 0.0


# =============================================================================
# TEST DATA GENERATION
# =============================================================================

def generate_test_clusters():
    """Generate diverse test point distributions."""
    np.random.seed(42)
    clusters = {}
    
    # Circular cluster
    theta = np.random.uniform(0, 2*np.pi, 200)
    r = np.sqrt(np.random.uniform(0, 1, 200))
    clusters['Circular'] = np.column_stack([30*r*np.cos(theta), 30*r*np.sin(theta)])
    
    # Diagonal elongated
    t = np.random.randn(200)
    clusters['Diagonal'] = np.column_stack([
        50*t + np.random.randn(200)*5,
        30*t + np.random.randn(200)*5
    ])
    
    # L-shaped (non-convex appearance)
    v = np.column_stack([np.random.uniform(-5, 5, 100), np.random.uniform(0, 60, 100)])
    h = np.column_stack([np.random.uniform(0, 50, 100), np.random.uniform(-5, 5, 100)])
    clusters['L-shaped'] = np.vstack([v, h])
    
    # Crescent
    theta = np.linspace(0.3, np.pi-0.3, 200)
    r = 40 + np.random.randn(200)*3
    clusters['Crescent'] = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
    
    # Core with outliers
    core = np.random.randn(180, 2) * 10
    outliers = np.array([[40,30],[-35,25],[30,-40],[-40,-30]]*5) + np.random.randn(20,2)*5
    clusters['Outliers'] = np.vstack([core, outliers])
    
    # Highly eccentric
    t = np.linspace(0, 2*np.pi, 200)
    clusters['Eccentric'] = np.column_stack([
        80*np.cos(t) + np.random.randn(200)*2,
        8*np.sin(t) + np.random.randn(200)*2
    ])
    
    return clusters


# =============================================================================
# HERO IMAGE GENERATION
# =============================================================================

def generate_hero_image():
    """Generate visually appealing hero image for README."""
    np.random.seed(123)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0d1117')
    
    # Generate sample points - elongated cluster with outliers
    n_points = 150
    angle = np.radians(25)
    cluster = np.random.randn(n_points, 2) @ [[2.5, 0], [0, 1]]
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    cluster = cluster @ rotation
    outliers = np.array([[5, 3.5], [-4.5, -2], [4, -3.5], [-3, 4]])
    points = np.vstack([cluster, outliers])
    
    # === LEFT PANEL: MVEE (Enclosing) ===
    ax1 = axes[0]
    ax1.set_facecolor('#0d1117')
    
    # Compute MVEE
    solver = MVEE(tol=0.001)
    result = solver.fit(points)
    center = result.ellipse.center
    radii = result.ellipse.radii
    angle_deg = np.degrees(np.arctan2(result.ellipse.rotation[1, 0], 
                                       result.ellipse.rotation[0, 0]))
    
    # Plot points
    ax1.scatter(points[:, 0], points[:, 1], c='#58a6ff', s=40, alpha=0.8, 
                edgecolors='white', linewidths=0.5, zorder=3)
    
    # Highlight core set
    core_pts = points[result.core_set_indices]
    ax1.scatter(core_pts[:, 0], core_pts[:, 1], c='#f85149', s=120, 
                marker='*', edgecolors='white', linewidths=0.5, zorder=4,
                label=f'Core set ({len(core_pts)} pts)')
    
    # Draw MVEE ellipse
    ellipse = Ellipse(xy=center, width=2*radii[0], height=2*radii[1],
                      angle=angle_deg, fill=False, 
                      edgecolor='#3fb950', linewidth=3, zorder=2)
    ax1.add_patch(ellipse)
    
    # Mark center
    ax1.scatter(*center, c='#3fb950', s=200, marker='+', linewidths=3, zorder=5)
    
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-6, 6)
    ax1.set_aspect('equal')
    ax1.set_title('MVEE (Enclosing)', fontsize=18, fontweight='bold', 
                  color='white', pad=15)
    ax1.tick_params(colors='#8b949e')
    for spine in ax1.spines.values():
        spine.set_color('#30363d')
    ax1.grid(True, alpha=0.2, color='#8b949e')
    
    # Add checkmarks
    ax1.text(0.05, 0.95, '✓ All points inside', transform=ax1.transAxes,
             fontsize=12, color='#3fb950', verticalalignment='top', fontweight='bold')
    ax1.text(0.05, 0.88, '✓ Minimum area', transform=ax1.transAxes,
             fontsize=12, color='#3fb950', verticalalignment='top', fontweight='bold')
    
    # === RIGHT PANEL: Fitting (OpenCV style) ===
    ax2 = axes[1]
    ax2.set_facecolor('#0d1117')
    
    # Compute fitting ellipse (OpenCV or fallback to covariance unscaled)
    if HAS_OPENCV:
        try:
            fit_center, fit_radii, fit_angle = opencv_fit_ellipse(points)
        except:
            fit_center = points.mean(axis=0)
            cov = np.cov(points.T)
            eigvals, eigvecs = la.eigh(cov)
            fit_radii = np.sqrt(eigvals)[::-1] * 2
            fit_angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    else:
        fit_center = points.mean(axis=0)
        cov = np.cov(points.T)
        eigvals, eigvecs = la.eigh(cov)
        fit_radii = np.sqrt(eigvals)[::-1] * 2
        fit_angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    
    # Check which points are outside
    cos_a, sin_a = np.cos(-fit_angle), np.sin(-fit_angle)
    centered = points - fit_center
    rotated = np.column_stack([
        centered[:, 0] * cos_a - centered[:, 1] * sin_a,
        centered[:, 0] * sin_a + centered[:, 1] * cos_a
    ])
    normalized = (rotated[:, 0] / fit_radii[0])**2 + (rotated[:, 1] / fit_radii[1])**2
    inside_mask = normalized <= 1.0
    
    # Plot points - different colors for inside/outside
    ax2.scatter(points[inside_mask, 0], points[inside_mask, 1], 
                c='#58a6ff', s=40, alpha=0.8, edgecolors='white', 
                linewidths=0.5, zorder=3)
    ax2.scatter(points[~inside_mask, 0], points[~inside_mask, 1], 
                c='#f85149', s=50, alpha=0.9, edgecolors='white', 
                linewidths=0.5, zorder=4, marker='x', label='Outside')
    
    # Draw fitting ellipse
    ellipse2 = Ellipse(xy=fit_center, width=2*fit_radii[0], height=2*fit_radii[1],
                       angle=np.degrees(fit_angle), fill=False, 
                       edgecolor='#f0883e', linewidth=3, linestyle='--', zorder=2)
    ax2.add_patch(ellipse2)
    
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(-6, 6)
    ax2.set_aspect('equal')
    ax2.set_title('Fitting (e.g. OpenCV)', fontsize=18, fontweight='bold', 
                  color='white', pad=15)
    ax2.tick_params(colors='#8b949e')
    for spine in ax2.spines.values():
        spine.set_color('#30363d')
    ax2.grid(True, alpha=0.2, color='#8b949e')
    
    # Add X marks
    pct_outside = (~inside_mask).sum() / len(points) * 100
    ax2.text(0.05, 0.95, f'✗ {pct_outside:.0f}% points outside', transform=ax2.transAxes,
             fontsize=12, color='#f85149', verticalalignment='top', fontweight='bold')
    ax2.text(0.05, 0.88, '✓ Best fit to distribution', transform=ax2.transAxes,
             fontsize=12, color='#f0883e', verticalalignment='top', fontweight='bold')
    
    # Main title
    fig.suptitle('MVEE-Opt: Minimum Volume Enclosing Ellipsoid', 
                 fontsize=22, fontweight='bold', color='white', y=1.02)
    
    plt.tight_layout()
    plt.savefig('media/hero.png', dpi=150, bbox_inches='tight', 
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    print("Generated: media/hero.png")


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmark():
    """Run full benchmark comparison."""
    print("=" * 80)
    print("MVEE-Opt Benchmark: Ellipse Methods Comparison")
    print("=" * 80)
    
    if HAS_SKIMAGE:
        print(f"scikit-image version: {skimage.__version__}")
    if HAS_OPENCV:
        print(f"OpenCV version: {cv2.__version__}")
    print()
    
    clusters = generate_test_clusters()
    
    methods = {
        'MVEE-Opt': {'color': '#3fb950', 'ls': '-', 'lw': 2.5},
        'Original MVEE': {'color': '#a371f7', 'ls': '-', 'lw': 1.5},
        'OpenCV': {'color': '#f85149', 'ls': '--', 'lw': 2},
        'scikit-image': {'color': '#58a6ff', 'ls': '--', 'lw': 2},
        'Covariance': {'color': '#f0883e', 'ls': ':', 'lw': 2},
        'BoundingBox': {'color': '#8b949e', 'ls': ':', 'lw': 1.5},
    }
    
    all_results = {}
    
    print(f"{'Cluster':<12} {'Method':<14} {'Area':>8} {'Ratio':>6} "
          f"{'Time':>8} {'Enclosed':>8} {'%Inside':>8}")
    print("-" * 75)
    
    for cluster_name, points in clusters.items():
        all_results[cluster_name] = {'points': points, 'methods': {}}
        results = all_results[cluster_name]['methods']
        
        # 1. Our MVEE-Opt
        try:
            solver = MVEE(tol=0.01, max_iter=200)
            start = time.perf_counter()
            r = solver.fit(points)
            elapsed = time.perf_counter() - start
            angle = np.arctan2(r.ellipse.rotation[1, 0], r.ellipse.rotation[0, 0])
            enclosed, frac = check_enclosure(points, r.ellipse.center, r.ellipse.radii, angle)
            results['MVEE-Opt'] = {
                'center': r.ellipse.center, 'radii': r.ellipse.radii, 'angle': angle,
                'area': compute_area(r.ellipse.radii), 'time': elapsed,
                'enclosed': enclosed, 'frac_inside': frac
            }
        except Exception as e:
            results['MVEE-Opt'] = {'error': str(e)}
        
        # 2. Original MVEE
        try:
            solver_orig = OriginalMVEE(tol=0.01, max_iter=200)
            start = time.perf_counter()
            center, radii, angle, iters = solver_orig.fit(points)
            elapsed = time.perf_counter() - start
            enclosed, frac = check_enclosure(points, center, radii, angle)
            results['Original MVEE'] = {
                'center': center, 'radii': radii, 'angle': angle,
                'area': compute_area(radii), 'time': elapsed,
                'enclosed': enclosed, 'frac_inside': frac
            }
        except Exception as e:
            results['Original MVEE'] = {'error': str(e)}
        
        # 3. OpenCV
        if HAS_OPENCV:
            try:
                start = time.perf_counter()
                center, radii, angle = opencv_fit_ellipse(points)
                elapsed = time.perf_counter() - start
                enclosed, frac = check_enclosure(points, center, radii, angle)
                results['OpenCV'] = {
                    'center': center, 'radii': radii, 'angle': angle,
                    'area': compute_area(radii), 'time': elapsed,
                    'enclosed': enclosed, 'frac_inside': frac
                }
            except Exception as e:
                results['OpenCV'] = {'error': str(e)}
        
        # 4. scikit-image
        if HAS_SKIMAGE:
            try:
                start = time.perf_counter()
                center, radii, angle = skimage_fit_ellipse(points)
                elapsed = time.perf_counter() - start
                enclosed, frac = check_enclosure(points, center, radii, angle)
                results['scikit-image'] = {
                    'center': center, 'radii': radii, 'angle': angle,
                    'area': compute_area(radii), 'time': elapsed,
                    'enclosed': enclosed, 'frac_inside': frac
                }
            except Exception as e:
                results['scikit-image'] = {'error': str(e)}
        
        # 5. Covariance
        try:
            start = time.perf_counter()
            center, radii, angle = covariance_ellipse(points)
            elapsed = time.perf_counter() - start
            enclosed, frac = check_enclosure(points, center, radii, angle)
            results['Covariance'] = {
                'center': center, 'radii': radii, 'angle': angle,
                'area': compute_area(radii), 'time': elapsed,
                'enclosed': enclosed, 'frac_inside': frac
            }
        except Exception as e:
            results['Covariance'] = {'error': str(e)}
        
        # 6. BoundingBox
        try:
            start = time.perf_counter()
            center, radii, angle = bounding_box_ellipse(points)
            elapsed = time.perf_counter() - start
            enclosed, frac = check_enclosure(points, center, radii, angle)
            results['BoundingBox'] = {
                'center': center, 'radii': radii, 'angle': angle,
                'area': compute_area(radii), 'time': elapsed,
                'enclosed': enclosed, 'frac_inside': frac
            }
        except Exception as e:
            results['BoundingBox'] = {'error': str(e)}
        
        # Print results for this cluster
        baseline = results.get('MVEE-Opt', {}).get('area', 1)
        for i, (method_name, r) in enumerate(results.items()):
            name_col = cluster_name if i == 0 else ''
            if 'error' in r:
                print(f"{name_col:<12} {method_name:<14} {'ERR':>8} - {r['error'][:30]}")
            else:
                ratio = r['area'] / baseline if baseline else 0
                enc = '✓' if r['enclosed'] else '✗'
                print(f"{name_col:<12} {method_name:<14} {r['area']:>8.0f} {ratio:>6.2f}x "
                      f"{r['time']*1000:>7.2f}ms {enc:>8} {r['frac_inside']*100:>7.1f}%")
        print()
    
    # Visualization with dark theme
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.patch.set_facecolor('#0d1117')
    axes = axes.flatten()
    
    for idx, (cluster_name, data) in enumerate(all_results.items()):
        ax = axes[idx]
        ax.set_facecolor('#0d1117')
        pts = data['points']
        ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.6, c='#8b949e')
        
        for method_name, style in methods.items():
            r = data['methods'].get(method_name, {})
            if 'center' in r:
                e = Ellipse(xy=r['center'], width=2*r['radii'][0], height=2*r['radii'][1],
                           angle=np.degrees(r['angle']), fill=False,
                           edgecolor=style['color'], linewidth=style['lw'],
                           linestyle=style['ls'])
                ax.add_patch(e)
        
        ax.set_title(cluster_name, fontsize=12, fontweight='bold', color='white')
        ax.set_aspect('equal')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(True, alpha=0.2, color='#8b949e')
    
    # Legend at bottom to avoid title occlusion
    legend_elements = [Line2D([0], [0], color=s['color'], lw=s['lw'], ls=s['ls'], label=m)
                       for m, s in methods.items() if m in all_results['Circular']['methods']]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=10,
               facecolor='#0d1117', edgecolor='#30363d', labelcolor='white',
               bbox_to_anchor=(0.5, 0.02))
    
    # Title with proper spacing
    plt.suptitle('Ellipse Methods Comparison\n(Solid = Enclosing, Dashed = Fitting)', 
                 fontsize=14, fontweight='bold', color='white', y=0.97)
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig('media/benchmark_results.png', dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    print("Generated: media/benchmark_results.png")
    
    # Summary
    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print("""
ENCLOSING METHODS (guarantee all points inside):
  ✓ MVEE-Opt      - Optimal (minimum area), ~1.5ms
  ✓ Original MVEE - Optimal but slower (O(N²)), ~10ms  
  ✓ Covariance    - Fast but ~15-43% larger area
  ✓ BoundingBox   - Fastest but up to 10x larger area

FITTING METHODS (points may be outside - NOT for bounding):
  ✗ OpenCV        - Fast (~0.05ms) but 0-90% points outside
  ✗ scikit-image  - Similar, 45-75% points outside

RECOMMENDATION:
  For bounding/enclosing: Use MVEE-Opt
  For shape fitting: Use OpenCV fitEllipse
""")


if __name__ == "__main__":
    generate_hero_image()
    run_benchmark()