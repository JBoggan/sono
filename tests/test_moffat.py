import numpy as np
import pytest
import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from sono_project.knot import Knot
from sono_project.sono_procedures import (
    control_leashes,
    find_neighbours,
    remove_overlaps,
    # Note: Moffat test might benefit from shift_nodes to break symmetry
    shift_nodes
)
from sono_project.knot_properties import (
    calculate_crossing_number_xy,
    calculate_writhe
)
# Import the helper function from the other test file (or move it to a shared conftest.py)
# For simplicity, let's copy it here for now, but refactoring later is good.
def run_sono_simplified(
    knot: Knot,
    max_iterations: int,
    num_of_it: int = 50,
    scaling_factor: float = 0.999, # Slower scaling might be needed
    overlap_threshold: float = 1e-5,
    delta: float = 0.01,
    epsilon: float = 0.1,
    skipped: int = 1,
    enable_shift_nodes: bool = False, # Add option for SN
    shift_freq: int = 20
    ) -> None:
    """Runs a simplified SONO loop for testing purposes."""
    neighbours_list = []
    print(f"\nRunning SONO (Moffat Test) for {max_iterations} iterations...")
    print(f"Initial state: {knot}, L/D={knot.length/knot.diameter:.2f}")
    initial_L = knot.length
    min_L_over_D = knot.length / knot.diameter

    for iteration in range(max_iterations):
        if iteration % num_of_it == 0:
            neighbours_list = find_neighbours(knot, skipped, epsilon)

        min_l, max_l = control_leashes(knot)
        max_ov, avg_ov = remove_overlaps(knot, neighbours_list, delta)

        if enable_shift_nodes and iteration % shift_freq == 0:
             shift_nodes(knot, 0.05) # Use a default shift fraction

        tightened_this_iter = False
        if avg_ov < overlap_threshold:
            knot.nodes *= scaling_factor
            knot.update_length()
            knot.target_leash_length *= scaling_factor
            tightened_this_iter = True

        # Track minimum L/D achieved
        current_L_over_D = knot.length / knot.diameter
        min_L_over_D = min(min_L_over_D, current_L_over_D)

        if iteration % (max_iterations // 20) == 0 or iteration == max_iterations - 1:
             acn = calculate_crossing_number_xy(knot)
             wr = calculate_writhe(knot)
             action = "Tightened" if tightened_this_iter else "Relaxed"
             print(f"Iter {iteration:6d}: L/D={current_L_over_D:.3f} (min={min_L_over_D:.3f}), AvgOv={avg_ov:.2e}, ACN={acn:.1f}, Wr={wr:.3f}, Action: {action}")

    final_lambda = knot.length / knot.diameter
    print(f"Final state: {knot}, L/D={final_lambda:.3f} (min={min_L_over_D:.3f})")

def generate_torus_knot_coords(p: int, q: int, R: float, r: float, num_points: int) -> np.ndarray:
    """
    Generates coordinates for a (p, q) torus knot.
    p, q are coprime integers.
    R is the major radius, r is the minor radius (R > r).
    """
    t = np.linspace(0, 2 * np.pi * q, num_points, endpoint=False) # Parameter runs 0 to 2*pi*q for full knot
    x = (R + r * np.cos(p * t / q)) * np.cos(t)
    y = (R + r * np.cos(p * t / q)) * np.sin(t)
    z = r * np.sin(p * t / q)
    return np.stack([x, y, z], axis=-1)

def test_moffat_t32_to_t23():
    """
    Test if SONO relaxes a T(3,2) trefoil configuration towards the T(2,3) form.
    Ref: Paper Section 4.2, Fig 3.
    We expect the ACN to stay ~3, Writhe ~ +/-3, and L/D to decrease.
    The paper shows fluctuations, potentially requiring ShiftNodes.
    """
    p, q = 3, 2
    R_major = 3.0
    r_minor = 1.0
    n_points = 100
    knot_diameter = 0.5 # Choose a reasonable diameter relative to radii

    initial_coords = generate_torus_knot_coords(p, q, R_major, r_minor, n_points)

    knot = Knot(coordinates=initial_coords, diameter=knot_diameter)

    initial_acn = calculate_crossing_number_xy(knot)
    initial_wr = calculate_writhe(knot)
    initial_L_over_D = knot.length / knot.diameter
    print(f"\nMoffat Test: Initial T({p},{q}) N={knot.num_nodes}, D={knot.diameter}")
    print(f"Initial L/D={initial_L_over_D:.3f}, ACN(xy)={initial_acn:.1f}, Wr={initial_wr:.3f}")

    # Assert initial properties are roughly correct for T(3,2) trefoil
    # ACN might not be exactly 3 in XY projection depending on initial view
    assert initial_acn >= 2.9, f"Initial ACN(xy) {initial_acn} is too low for a trefoil"
    assert abs(initial_wr) > 1.0, f"Initial Writhe {initial_wr} is too low for a trefoil"

    # Determine initial 'skipped' based on average leash length
    l_avg = knot.length / knot.num_nodes
    # Adjust skipped calculation based on paper suggestion (pi*D / 2l)
    skipped_val = max(1, int(round(np.pi * knot.diameter / (2 * l_avg))))
    print(f"Using skipped={skipped_val}")

    # Run the simplified SONO process
    # Might need more iterations and ShiftNodes enabled based on paper description
    run_sono_simplified(
        knot,
        max_iterations=10000, # Increased iterations
        num_of_it=100,      # Less frequent FN
        scaling_factor=0.999, # Slower scaling
        overlap_threshold=1e-5,
        delta=0.005,
        epsilon=0.1 * knot_diameter,
        skipped=skipped_val,
        enable_shift_nodes=True, # Enable SN as per paper observation
        shift_freq=10          # Frequent shifts might help break symmetry
    )

    # Assertions: Check if the knot simplified and maintained topology
    final_acn = calculate_crossing_number_xy(knot)
    final_wr = calculate_writhe(knot)
    final_L_over_D = knot.length / knot.diameter

    print(f"Final L/D={final_L_over_D:.3f}, ACN(xy)={final_acn:.1f}, Wr={final_wr:.3f}")

    # Check topology hasn't changed drastically (still a trefoil)
    assert final_acn >= 2.9, f"Final ACN(xy) {final_acn} is too low for a trefoil"
    assert abs(final_wr) > 1.0, f"Final Writhe {final_wr} is too low for a trefoil"

    # Check if L/D decreased from the *initial* value significantly.
    # The paper's graph shows initial tightening then relaxation.
    # Check against initial L/D, not necessarily the lowest point reached during fluctuation.
    assert final_L_over_D < initial_L_over_D * 0.95, f"L/D did not decrease significantly ({initial_L_over_D:.3f} -> {final_L_over_D:.3f})"

    # A more advanced test could involve comparing the final geometry to
    # an ideal T(2,3) structure, but that's harder to define and compare automatically. 