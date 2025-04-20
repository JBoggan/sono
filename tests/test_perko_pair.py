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
    shift_nodes,
    normalize_node_number # Normalization might be useful here
)
from sono_project.knot_properties import (
    calculate_crossing_number_xy,
    calculate_writhe
)

# --- Helper Functions ---
# For simplicity, copy run_sono_simplified here. Refactor later.
def run_sono_simplified(
    knot: Knot,
    max_iterations: int,
    num_of_it: int = 50,
    scaling_factor: float = 0.999,
    overlap_threshold: float = 1e-5,
    delta: float = 0.01,
    epsilon: float = 0.1,
    skipped: int = 1,
    enable_shift_nodes: bool = True,
    shift_freq: int = 10,
    enable_normalize: bool = False,
    normalize_freq: int = 1000,
    normalize_density: float = 10.0
    ) -> None:
    """Runs a simplified SONO loop for testing purposes."""
    neighbours_list = []
    test_name = "Perko Pair Test" # Customize print slightly
    print(f"\nRunning SONO ({test_name}) for {max_iterations} iterations...")
    print(f"Initial state: {knot}, L/D={knot.length/knot.diameter:.2f}")
    initial_L = knot.length
    min_L_over_D = knot.length / knot.diameter

    for iteration in range(max_iterations):
        recalc_neighbors = False
        if iteration % num_of_it == 0:
            neighbours_list = find_neighbours(knot, skipped, epsilon)

        min_l, max_l = control_leashes(knot)
        max_ov, avg_ov = remove_overlaps(knot, neighbours_list, delta)

        if enable_shift_nodes and iteration % shift_freq == 0:
             shift_nodes(knot, 0.05)

        tightened_this_iter = False
        if avg_ov < overlap_threshold:
            knot.nodes *= scaling_factor
            knot.update_length()
            knot.target_leash_length *= scaling_factor
            tightened_this_iter = True

        if enable_normalize and iteration > 0 and iteration % normalize_freq == 0:
            normalize_node_number(knot, target_density=normalize_density)
            # Need to recalculate neighbors immediately after normalization
            neighbours_list = find_neighbours(knot, skipped, epsilon)
            recalc_neighbors = True

        # Track minimum L/D achieved
        current_L_over_D = knot.length / knot.diameter
        min_L_over_D = min(min_L_over_D, current_L_over_D)

        if iteration % (max_iterations // 10) == 0 or iteration == max_iterations - 1:
             acn = calculate_crossing_number_xy(knot)
             wr = calculate_writhe(knot)
             action = "Tightened" if tightened_this_iter else ("Normalized" if recalc_neighbors else "Relaxed")
             print(f"Iter {iteration:6d}: L/D={current_L_over_D:.3f} (min={min_L_over_D:.3f}), N={knot.num_nodes}, AvgOv={avg_ov:.2e}, ACN={acn:.1f}, Wr={wr:.3f}, Action: {action}")

    final_lambda = knot.length / knot.diameter
    print(f"Final state: {knot}, L/D={final_lambda:.3f} (min={min_L_over_D:.3f})")

# --- Perko Pair Initial Coordinates ---
# Representative coordinates for the two forms (e.g., adapted from online sources)
# These are just examples, might need adjustment or different sources.
# Source inspiration: KnotAtlas, KnotPlot, etc.

# Perko Pair A (10_161 form)
coords_A = np.array([
    [0.8, -1. , -0.5],
    [1. , -0.5, -0.8],
    [0.5, 0. , -1. ],
    [0. , 0.5, -0.8],
    [-0.8, 1. , -0.5],
    [-1. , 0.8, 0. ],
    [-0.5, 1. , 0.5],
    [0. , 1. , 0.8],
    [0.5, 0.8, 1. ],
    [1. , 0. , 0.5],
    [0.8, -0.5, 0. ],
    [0. , -1. , 0. ],
    [-0.8, -0.8, -0.5],
    [-1. , 0. , -0.8],
    [-0.5, 0.5, -1. ],
    [0. , 0. , -0.5],
    [0. , -0.5, 0. ],
    [0.5, -0.8, 0.5],
    [0.8, 0. , 0.8],
    [0. , 0.5, 1. ]
]) * 3.0 # Scale up

# Perko Pair B (10_162 form) - Should look different initially
coords_B = np.array([
    [ 1. , -0.5, -0.8],
    [ 0.5, 0. , -1. ],
    [ -0.5, 0. , -1. ],
    [ -1. , -0.5, -0.8],
    [ -0.8, -1. , -0.5],
    [ 0. , -1. , 0. ],
    [ 0.8, -1. , -0.5],
    [ 1. , -0.8, 0. ],
    [ 0.5, -0.5, 0.5],
    [ 0. , 0. , 0.8],
    [ -0.5, -0.5, 0.5],
    [ -1. , -0.8, 0. ],
    [ -0.8, 0. , 0.5],
    [ 0. , 0.5, 0.8],
    [ 0.8, 0. , 0.5],
    [ 1. , 0.5, 0. ],
    [ 0.5, 0.8, -0.5],
    [ 0. , 1. , -0.8],
    [ -0.5, 0.8, -0.5],
    [ -1. , 0.5, 0. ]
]) * 3.0 # Scale up


def test_perko_pair_convergence():
    """
    Test if SONO brings the two Perko pair conformations (10_161, 10_162)
    to the same final state (same L/D, ACN, Wr).
    Ref: Paper Section 4.3, Fig 4.
    """
    knot_diameter = 0.6 # Needs careful selection relative to coordinates
    num_points_target = 100 # Target for normalization

    # Initialize Knot A
    knot_A = Knot(coordinates=coords_A, diameter=knot_diameter)
    normalize_node_number(knot_A, target_density=num_points_target / (knot_A.length / knot_A.diameter))
    print("\n--- Initializing Perko Knot A (10_161 form) ---")
    print(knot_A)
    l_avg_A = knot_A.length / knot_A.num_nodes
    skipped_A = max(1, int(round(np.pi * knot_A.diameter / (2 * l_avg_A))))

    # Initialize Knot B
    knot_B = Knot(coordinates=coords_B, diameter=knot_diameter)
    normalize_node_number(knot_B, target_density=num_points_target / (knot_B.length / knot_B.diameter))
    print("\n--- Initializing Perko Knot B (10_162 form) ---")
    print(knot_B)
    l_avg_B = knot_B.length / knot_B.num_nodes
    skipped_B = max(1, int(round(np.pi * knot_B.diameter / (2 * l_avg_B))))

    # Ensure initial ACN/Wr are somewhat different if possible (depends on projection)
    acn_A_init = calculate_crossing_number_xy(knot_A)
    wr_A_init = calculate_writhe(knot_A)
    acn_B_init = calculate_crossing_number_xy(knot_B)
    wr_B_init = calculate_writhe(knot_B)
    print(f"Knot A Initial: ACN={acn_A_init:.1f}, Wr={wr_A_init:.3f}")
    print(f"Knot B Initial: ACN={acn_B_init:.1f}, Wr={wr_B_init:.3f}")

    # Define common SONO parameters
    sono_params = {
        "max_iterations": 15000, # May need more iterations
        "num_of_it": 100,
        "scaling_factor": 0.999,
        "overlap_threshold": 1e-5,
        "delta": 0.005,
        "epsilon": 0.1 * knot_diameter,
        "enable_shift_nodes": True,
        "shift_freq": 10,
        "enable_normalize": True, # Normalization helps compare final states
        "normalize_freq": 2000,
        "normalize_density": 10.0 
    }

    # Run SONO on Knot A
    print("\n--- Running SONO on Knot A --- ")
    run_sono_simplified(knot_A, skipped=skipped_A, **sono_params)

    # Run SONO on Knot B
    print("\n--- Running SONO on Knot B --- ")
    run_sono_simplified(knot_B, skipped=skipped_B, **sono_params)

    # --- Assertions --- 
    final_L_A = knot_A.length
    final_D_A = knot_A.diameter
    final_acn_A = calculate_crossing_number_xy(knot_A)
    final_wr_A = calculate_writhe(knot_A)

    final_L_B = knot_B.length
    final_D_B = knot_B.diameter
    final_acn_B = calculate_crossing_number_xy(knot_B)
    final_wr_B = calculate_writhe(knot_B)

    print("\n--- Final Comparison ---")
    print(f"Knot A Final: L/D={final_L_A/final_D_A:.3f}, ACN={final_acn_A:.1f}, Wr={final_wr_A:.3f}, N={knot_A.num_nodes}")
    print(f"Knot B Final: L/D={final_L_B/final_D_B:.3f}, ACN={final_acn_B:.1f}, Wr={final_wr_B:.3f}, N={knot_B.num_nodes}")

    # Check if the key scalar invariants converged to the same values
    # Use np.isclose for floating point comparison with tolerance
    atol = 1e-2 # Absolute tolerance for L/D and Wr
    rtol = 1e-2 # Relative tolerance

    assert np.isclose(final_L_A / final_D_A, final_L_B / final_D_B, rtol=rtol, atol=atol), \
           f"Final L/D values differ significantly: {final_L_A/final_D_A:.4f} vs {final_L_B/final_D_B:.4f}"
    assert np.isclose(final_acn_A, final_acn_B, atol=0.1), \
           f"Final ACN(xy) values differ: {final_acn_A:.1f} vs {final_acn_B:.1f}"
    assert np.isclose(final_wr_A, final_wr_B, rtol=rtol, atol=atol), \
           f"Final Writhe values differ significantly: {final_wr_A:.4f} vs {final_wr_B:.4f}"

    print("\nPerko Pair Test Passed: Final scalar invariants (L/D, ACN, Wr) converged.")

    # Advanced check: Geometric similarity (RMSD after alignment)
    # This requires an alignment algorithm (e.g., Kabsch)
    # For now, we rely on the scalar invariants as the primary check. 