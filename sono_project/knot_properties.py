import numpy as np
from typing import Optional

try:
    from .knot import Knot
except ImportError:
    from knot import Knot

# --- Crossing Number (ACN Placeholder) ---

def _orientation(p, q, r):
    """Return orientation of ordered triplet (p, q, r)."""
    val = (q[1] - p[1]) * (r[0] - q[0]) - \
          (q[0] - p[0]) * (r[1] - q[1])
    if np.abs(val) < 1e-9: return 0 # Collinear
    return 1 if val > 0 else -1 # Clockwise or Counterclockwise

def _on_segment(p, q, r):
    """Check if point q lies on segment pr."""
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

def _segments_intersect(p1, q1, p2, q2):
    """Check if line segment 'p1q1' and 'p2q2' intersect."""
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    # General case
    if o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0:
        if o1 != o2 and o3 != o4:
            return True
        else:
            return False # Parallel non-overlapping

    # Special Cases (Collinear)
    # Check only if endpoints lie on the other segment if orientation is 0
    if o1 == 0 and _on_segment(p1, p2, q1): return True
    if o2 == 0 and _on_segment(p1, q2, q1): return True
    if o3 == 0 and _on_segment(p2, p1, q2): return True
    if o4 == 0 and _on_segment(p2, q1, q2): return True

    return False # Doesn't intersect

def calculate_crossing_number_xy(knot: Knot) -> float:
    """
    Calculates the number of crossings in the projection onto the xy-plane.

    NOTE: This is a simplified version. True ACN involves averaging over
    many projection directions. This serves as a placeholder based on the paper.

    Args:
        knot: The Knot object.

    Returns:
        The number of crossings in the xy-projection.
    """
    n = knot.num_nodes
    if n < 4:
        return 0.0

    nodes_2d = knot.nodes[:, :2] # Project onto xy-plane
    crossing_count = 0

    for i in range(n):
        p1 = nodes_2d[i]
        q1 = nodes_2d[(i + 1) % n]

        # Check against non-adjacent segments (k starts from i+2)
        for k in range(i + 2, n):
            # Avoid adjacent or coincident segments
            if (k + 1) % n == i:
                continue

            p2 = nodes_2d[k]
            q2 = nodes_2d[(k + 1) % n]

            if _segments_intersect(p1, q1, p2, q2):
                crossing_count += 1

    # Each crossing involves two segments, intersection check finds each once.
    return float(crossing_count)

# --- Writhe (Wr) ---

def calculate_writhe(knot: Knot) -> float:
    """
    Calculates the writhe of the knot using a discrete approximation
    of the Gauss linking integral.

    Formula used:
    Wr = (1 / 4pi) * sum_{i=0}^{N-1} sum_{k=i+1}^{N-1} Omega(i, k)
    where Omega is the solid angle contribution from segments i and k.

    Args:
        knot: The Knot object.

    Returns:
        The calculated writhe.
    """
    n = knot.num_nodes
    if n < 3:
        return 0.0

    nodes = knot.nodes
    total_writhe_term = 0.0
    epsilon = 1e-9 # To avoid division by zero

    # Define segments vectors more clearly
    segments = np.diff(nodes, axis=0, append=nodes[0:1])

    for i in range(n):
        # Segment i: nodes[i] to nodes[(i+1)%n]
        ri = nodes[i]
        vi = segments[i]

        for k in range(i + 1, n):
            # Segment k: nodes[k] to nodes[(k+1)%n]
            # Ensure segments are not adjacent (already handled by k=i+1 start?)
            # Need to be careful if i=N-1, k=0 etc. No, loop structure prevents this.
            # The formula uses r_i and r_k as points on the curve.
            # Let's use segment midpoints for r_i and r_k to be safer?
            # Or stick to the formula using node indices?
            # Paper isn't specific. Let's use node indices r_i, r_k and
            # segment vectors v_i, v_k.

            rk = nodes[k]
            vk = segments[k] # vk = nodes[(k+1)%n] - nodes[k]

            rik = ri - rk
            dist_rik_sq = np.dot(rik, rik)

            if dist_rik_sq < epsilon:
                continue # Avoid division by zero if nodes are too close

            dist_rik = np.sqrt(dist_rik_sq)

            # Calculate the term inside the sum: (v_i x v_k) . (r_i - r_k) / |r_i - r_k|^3
            cross_prod = np.cross(vi, vk)
            dot_prod = np.dot(cross_prod, rik)

            term = dot_prod / (dist_rik ** 3)
            total_writhe_term += term

    writhe = total_writhe_term / (4 * np.pi)
    return writhe

# Example usage
if __name__ == '__main__':
    # Simple square knot (expected Wr=0, ACN=0)
    coords_square = np.array([
        [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]
    ])
    knot_square = Knot(coordinates=coords_square, diameter=0.1)
    print(f"Square: {knot_square}")
    acn_sq = calculate_crossing_number_xy(knot_square)
    wr_sq = calculate_writhe(knot_square)
    print(f"  ACN(xy) = {acn_sq}, Writhe = {wr_sq:.4f}")

    # Figure-eight knot example (non-trivial)
    # Coordinates from KnotPlot (simplified)
    coords_fig8 = np.array([
        [ 2.,  0.,  1.], [ 1., -1.,  0.], [ 0.,  0., -1.],
        [-1.,  1.,  0.], [-2.,  0.,  1.], [-1., -1.,  0.],
        [ 0.,  0., -1.], [ 1.,  1.,  0.]
    ]) * 0.5 # Scale it down a bit
    knot_fig8 = Knot(coordinates=coords_fig8, diameter=0.1)
    print(f"\nFigure 8: {knot_fig8}")
    acn_f8 = calculate_crossing_number_xy(knot_fig8)
    wr_f8 = calculate_writhe(knot_fig8)
    print(f"  ACN(xy) = {acn_f8}, Writhe = {wr_f8:.4f}") # Expected ACN=4, Wr=0 for ideal figure 8

    # Simple Trefoil (3_1) (non-trivial writhe)
    t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    coords_trefoil = np.stack([x, y, z], axis=-1) * 0.2
    knot_trefoil = Knot(coordinates=coords_trefoil, diameter=0.1)
    print(f"\nTrefoil (approx): {knot_trefoil}")
    acn_tf = calculate_crossing_number_xy(knot_trefoil)
    wr_tf = calculate_writhe(knot_trefoil)
    print(f"  ACN(xy) = {acn_tf}, Writhe = {wr_tf:.4f}") # Expected ACN=3, Wr approx +/- 3 