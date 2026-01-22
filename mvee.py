"""
MVEE-Opt: Fast Minimum Volume Enclosing Ellipsoid
==================================================

A production-ready Python implementation of the Minimum Volume Enclosing 
Ellipsoid (MVEE) algorithm, also known as the Löwner-John ellipsoid.

Uses Khachiyan's dual formulation with:
- Einsum-accelerated leverage score computation (O(Nd²) vs O(N²d))
- Convex hull preprocessing for 2D point sets
- Dual convergence criteria (KKT optimality + objective stagnation)
- Guaranteed enclosure via post-hoc verification

Author: TonyLuna
Date: January 21, 2026
License: MIT

Usage:
    from mvee import MVEE
    
    points = np.random.randn(100, 2)
    result = MVEE().fit(points)
    
    print(result.ellipse.center)
    print(result.ellipse.radii)

References:
    - Khachiyan (1996): Rounding of polytopes
    - Todd & Yıldırım (2007): On Khachiyan's algorithm for MVEE
    - Kumar & Yıldırım (2005): MVEE and core sets
"""

import numpy as np
from numpy import linalg as la
from typing import NamedTuple
from dataclasses import dataclass

__version__ = "1.0.0"
__all__ = ["MVEE", "MVEEResult", "EllipseParams"]


class EllipseParams(NamedTuple):
    """Geometric parameters of the enclosing ellipsoid.
    
    Attributes:
        center: Centroid of the ellipsoid (d,)
        radii: Semi-axis lengths, major axis first (d,)
        rotation: Orthogonal rotation matrix (d, d)
        A: Shape matrix for constraint (x-c)ᵀA(x-c) ≤ 1
    """
    center: np.ndarray
    radii: np.ndarray
    rotation: np.ndarray
    A: np.ndarray


@dataclass
class MVEEResult:
    """Result of MVEE computation.
    
    Attributes:
        ellipse: Geometric parameters (center, radii, rotation, A)
        core_set_indices: Indices of points on ellipsoid boundary
        weights: Dual weight vector (non-zero = boundary points)
        iterations: Number of iterations executed
        converged: True if algorithm converged before max_iter
    """
    ellipse: EllipseParams
    core_set_indices: np.ndarray
    weights: np.ndarray
    iterations: int
    converged: bool


class MVEE:
    """Minimum Volume Enclosing Ellipsoid solver.
    
    Computes the smallest ellipsoid containing all input points using
    Khachiyan's dual formulation with optimized leverage computation.
    
    Args:
        tol: Convergence tolerance (default: 0.01)
        max_iter: Maximum iterations (default: 200)
        use_convex_hull: Apply convex hull preprocessing for 2D (default: True)
    
    Example:
        >>> solver = MVEE(tol=0.001)
        >>> result = solver.fit(points)
        >>> print(f"Center: {result.ellipse.center}")
        >>> print(f"Converged in {result.iterations} iterations")
    """
    
    def __init__(self, tol: float = 0.01, max_iter: int = 200,
                 use_convex_hull: bool = True):
        self.tol = tol
        self.max_iter = max_iter
        self.use_convex_hull = use_convex_hull
    
    def fit(self, P: np.ndarray) -> MVEEResult:
        """Compute the minimum volume enclosing ellipsoid.
        
        Args:
            P: Point set of shape (N, d) where N ≥ d+1
        
        Returns:
            MVEEResult containing ellipse parameters and diagnostics
        
        Raises:
            ValueError: If fewer than d+1 points provided
        """
        P = np.asarray(P, dtype=np.float64)
        N, d = P.shape
        
        if N < d + 1:
            raise ValueError(f"Need at least {d+1} points for {d}D, got {N}")
        
        original_indices = np.arange(N)
        P_work = P.copy()
        
        # Convex hull preprocessing (2D only)
        if self.use_convex_hull and d == 2 and N > 10:
            hull_idx = self._convex_hull_2d(P)
            P_work = P[hull_idx]
            original_indices = original_indices[hull_idx]
        
        # Run Khachiyan iteration
        center, A, weights, iterations, converged = self._khachiyan(P_work, d)
        
        # Ensure enclosure of ALL original points (not just hull)
        centered = P - center
        distances = np.sum((centered @ A) * centered, axis=1)
        max_dist = np.max(distances)
        if max_dist > 1.0:
            A = A / (max_dist * 1.001)
        
        # Extract geometric parameters
        ellipse = self._extract_params(center, A)
        
        # Identify core set (boundary points)
        core_mask = weights > 1e-5
        core_indices = original_indices[core_mask]
        
        return MVEEResult(ellipse, core_indices, weights, iterations, converged)
    
    def _convex_hull_2d(self, P: np.ndarray) -> np.ndarray:
        """Graham scan convex hull for 2D points.
        
        Args:
            P: Points of shape (N, 2)
        
        Returns:
            Indices of convex hull vertices in counter-clockwise order
        """
        N = len(P)
        start = min(range(N), key=lambda i: (P[i, 1], P[i, 0]))
        
        def polar_key(idx):
            if idx == start:
                return (-np.inf, 0)
            dx, dy = P[idx] - P[start]
            return (np.arctan2(dy, dx), dx*dx + dy*dy)
        
        indices = sorted(range(N), key=polar_key)
        
        def cross(o, a, b):
            return ((P[a, 0] - P[o, 0]) * (P[b, 1] - P[o, 1]) - 
                    (P[a, 1] - P[o, 1]) * (P[b, 0] - P[o, 0]))
        
        hull = []
        for idx in indices:
            while len(hull) > 1 and cross(hull[-2], hull[-1], idx) <= 0:
                hull.pop()
            hull.append(idx)
        
        return np.array(hull)
    
    def _khachiyan(self, P: np.ndarray, d: int):
        """Khachiyan's algorithm for MVEE dual problem.
        
        Args:
            P: Working point set (N, d)
            d: Dimension
        
        Returns:
            Tuple of (center, A, weights, iterations, converged)
        """
        N = len(P)
        d1 = d + 1
        
        # Lift to homogeneous coordinates
        Q = np.hstack([P, np.ones((N, 1))])
        u = np.full(N, 1.0 / N)
        
        converged = False
        prev_obj = 0
        
        for iteration in range(1, self.max_iter + 1):
            # Weighted outer product matrix V = QᵀDQ where D = diag(u)
            X = Q * np.sqrt(u)[:, np.newaxis]
            V = X.T @ X
            
            # Robust matrix inversion
            try:
                V_inv = la.inv(V)
            except la.LinAlgError:
                V_inv = la.pinv(V)
            
            # Leverage scores: M[i] = Qᵢ · V⁻¹ · Qᵢᵀ (diagonal only!)
            M = np.einsum('ij,jk,ik->i', Q, V_inv, Q)
            
            # Find maximally violating point
            j = np.argmax(M)
            max_M = M[j]
            
            # Convergence check 1: KKT optimality
            if max_M <= d1 * (1 + self.tol):
                converged = True
                break
            
            # Convergence check 2: Objective stagnation
            det_val = la.det(V)
            obj = np.log(det_val) if det_val > 0 else prev_obj
            if iteration > 10:
                obj_change = abs(obj - prev_obj) / (abs(prev_obj) + 1e-10)
                if obj_change < self.tol * 0.01:
                    converged = True
                    break
            prev_obj = obj
            
            # Weight update
            step = (max_M - d1) / (d1 * (max_M - 1))
            step = np.clip(step, 1e-10, 0.5)
            u = (1 - step) * u
            u[j] += step
        
        # Extract ellipsoid parameters from dual solution
        center = P.T @ u
        P_centered = P - center
        Sigma = (P_centered.T * u) @ P_centered
        
        try:
            A = la.inv(Sigma) / d
        except la.LinAlgError:
            A = la.pinv(Sigma) / d
        
        return center, A, u, iteration, converged
    
    def _extract_params(self, center: np.ndarray, A: np.ndarray) -> EllipseParams:
        """Extract geometric parameters from shape matrix A.
        
        Args:
            center: Ellipsoid center
            A: Positive definite shape matrix
        
        Returns:
            EllipseParams with center, radii, rotation, A
        """
        eigenvalues, eigenvectors = la.eigh(A)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        radii = 1.0 / np.sqrt(eigenvalues)
        
        # Sort by major axis first
        order = np.argsort(radii)[::-1]
        radii = radii[order]
        rotation = eigenvectors[:, order]
        
        return EllipseParams(center, radii, rotation, A)


# Convenience function for simple usage
def mvee(points: np.ndarray, tol: float = 0.01) -> tuple:
    """Convenience function for quick MVEE computation.
    
    Args:
        points: Point set of shape (N, d)
        tol: Convergence tolerance
    
    Returns:
        Tuple of (center, radii, rotation_matrix)
    
    Example:
        >>> center, radii, rotation = mvee(points)
    """
    result = MVEE(tol=tol).fit(points)
    return result.ellipse.center, result.ellipse.radii, result.ellipse.rotation


if __name__ == "__main__":
    # Quick demo
    np.random.seed(42)
    points = np.random.randn(200, 2) * [3, 1]  # Elongated cluster
    
    solver = MVEE()
    result = solver.fit(points)
    
    print("MVEE-Opt Demo")
    print("=" * 40)
    print(f"Points: {len(points)}")
    print(f"Center: {result.ellipse.center}")
    print(f"Radii: {result.ellipse.radii}")
    print(f"Core set size: {len(result.core_set_indices)}")
    print(f"Iterations: {result.iterations}")
    print(f"Converged: {result.converged}")