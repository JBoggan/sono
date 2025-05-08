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
    normalize_node_number
)
from sono_project.knot_properties import (
    calculate_crossing_number_xy,
    calculate_writhe
)
# Import I/O and visualization
from sono_project.io_utils import write_knot_to_file
from sono_project.visualization import plot_knot

# --- Helper Functions ---
# For simplicity, copy run_sono_simplified here. Refactor later.
def run_sono_simplified(
    knot: Knot,
    max_iterations: int,
    num_of_it: int = 50,
    scaling_factor: float = 0.999,
    overlap_threshold: float = 1e-5,
    delta: float = 0.005, # Starting delta
    epsilon: float = 0.1,
    skipped: int = 1,
    enable_shift_nodes: bool = True,
    shift_freq: int = 10,
    enable_normalize: bool = False,
    normalize_freq: int = 1000,
    normalize_density: float = 10.0,
    # Parameters for potential temporary changes
    increase_delta_iter: int = -1, # Iteration to increase delta (-1 to disable)
    delta_increase_factor: float = 10.0,
    delta_increase_duration: int = 500 # How long to keep delta increased
    ) -> float:
    """Runs a simplified SONO loop for testing purposes. Returns min L/D found."""
    neighbours_list = []
    test_name = "Symmetry Breaking Test" # Customize print
    print(f"\nRunning SONO ({test_name}) for {max_iterations} iterations...")
    print(f"Initial state: {knot}, L/D={knot.length/knot.diameter:.2f}")
    min_L_over_D = knot.length / knot.diameter if knot.diameter > 0 else float('inf')
    original_delta = delta
    delta_end_iter = -1

    for iteration in range(max_iterations):
        # --- Parameter adjustments for symmetry breaking ---
        current_delta = delta
        if iteration == increase_delta_iter:
            current_delta = original_delta * delta_increase_factor
            delta_end_iter = iteration + delta_increase_duration
            print(f"*** Iter {iteration}: Temporarily increasing delta to {current_delta:.4f} for {delta_increase_duration} iterations ***")
        elif iteration == delta_end_iter:
            current_delta = original_delta # Restore delta
            print(f"*** Iter {iteration}: Restoring delta to {current_delta:.4f} ***")
        elif increase_delta_iter < iteration < delta_end_iter:
            current_delta = original_delta * delta_increase_factor
        else:
            current_delta = original_delta
        # -------------------------------------------------

        recalc_neighbors = False
        # Recalculate skipped dynamically if necessary *before* FN call
        # (Assuming dynamic skipping isn't explicitly controlled by args here)
        l_curr = knot.target_leash_length
        current_skipped = max(1, int(round(np.pi * knot.diameter / (2 * l_curr)))) if l_curr > 1e-9 else skipped

        if iteration % num_of_it == 0:
            neighbours_list = find_neighbours(knot, current_skipped, epsilon)

        min_l, max_l = control_leashes(knot)
        # Use the potentially modified delta for this iteration
        max_ov, avg_ov = remove_overlaps(knot, neighbours_list, current_delta)

        if enable_shift_nodes and iteration % shift_freq == 0:
             shift_nodes(knot, 0.05)

        tightened_this_iter = False
        if avg_ov < overlap_threshold:
            knot.nodes *= scaling_factor
            knot.update_length()
            knot.target_leash_length *= scaling_factor
            tightened_this_iter = True

        if enable_normalize and iteration > 0 and iteration % normalize_freq == 0:
            # Need to handle skipped update around normalization properly if used
            l_pre_norm = knot.target_leash_length
            skipped_pre_norm = max(1, int(round(np.pi * knot.diameter / (2 * l_pre_norm)))) if l_pre_norm > 1e-9 else current_skipped
            normalize_node_number(knot, target_density=normalize_density)
            neighbours_list = find_neighbours(knot, skipped_pre_norm, epsilon)
            recalc_neighbors = True

        current_L_over_D = knot.length / knot.diameter if knot.diameter > 0 else float('inf')
        min_L_over_D = min(min_L_over_D, current_L_over_D)

        if iteration % (max_iterations // 20) == 0 or iteration == max_iterations - 1:
             acn = calculate_crossing_number_xy(knot)
             wr = calculate_writhe(knot)
             action = "Tightened" if tightened_this_iter else ("Normalized" if recalc_neighbors else ("DeltaUp" if current_delta != original_delta else "Relaxed"))
             print(f"Iter {iteration:6d}: L/D={current_L_over_D:.3f} (min={min_L_over_D:.3f}), N={knot.num_nodes}, AvgOv={avg_ov:.2e}, ACN={acn:.1f}, Wr={wr:.3f}, Action: {action}")

    final_lambda = knot.length / knot.diameter if knot.diameter > 0 else float('inf')
    print(f"Final state: {knot}, L/D={final_lambda:.3f} (min={min_L_over_D:.3f})")
    return min_L_over_D # Return the minimum L/D reached during the run

# Copy from test_moffat.py (Refactor later)
def generate_torus_knot_coords(p: int, q: int, R: float, r: float, num_points: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi * q, num_points, endpoint=False)
    x = (R + r * np.cos(p * t / q)) * np.cos(t)
    y = (R + r * np.cos(p * t / q)) * np.sin(t)
    z = r * np.sin(p * t / q)
    return np.stack([x, y, z], axis=-1)

def test_t25_symmetry_breaking():
    """
    Test symmetry breaking for the T(2,5) knot (5_1 knot).
    Starts with a symmetric torus knot config, expects SONO to find a lower L/D
    asymmetrical state, possibly requiring parameter adjustment (increased delta).
    Ref: Paper Section 5, Fig 5.
    Symmetric L/D ~ 24.2, Asymmetric L/D ~ 23.5
    """
    # --- Test Setup ---
    output_dir = os.path.join(project_root, "test_outputs")
    initial_coords_filename = os.path.join(output_dir, "symbreak_initial_coords.txt")
    initial_plot_filename = os.path.join(output_dir, "symbreak_initial_plot.png")
    final_coords_filename = os.path.join(output_dir, "symbreak_final_coords.txt")
    final_plot_filename = os.path.join(output_dir, "symbreak_final_plot.png")
    os.makedirs(output_dir, exist_ok=True)

    p, q = 2, 5 # T(2,5) -> 5_1 knot
    R_major = 3.0
    r_minor = 1.0
    n_points = 150 # Use a reasonable number of points
    knot_diameter = 0.5
    initial_coords = generate_torus_knot_coords(p, q, R_major, r_minor, n_points)
    knot = Knot(coordinates=initial_coords, diameter=knot_diameter)

    # --- Save Initial State ---
    print(f"\nSaving initial state for Symmetry Breaking test...")
    try:
        write_knot_to_file(knot, initial_coords_filename)
        plot_knot(knot, title=f"Symmetry Breaking Test - Initial State T({p},{q})", save_path=initial_plot_filename, show_plot=False)
    except Exception as e:
        pytest.fail(f"Error during initial output saving for Symmetry Breaking test: {e}")

    initial_acn = calculate_crossing_number_xy(knot)
    initial_wr = calculate_writhe(knot)
    initial_L_over_D = knot.length / knot.diameter if knot.diameter > 0 else float('inf')
    print(f"\nSymmetry Breaking Test: Initial T({p},{q}) N={knot.num_nodes}, D={knot.diameter}")
    print(f"Initial L/D={initial_L_over_D:.3f}, ACN(xy)={initial_acn:.1f}, Wr={initial_wr:.3f}")

    assert initial_acn >= 4.9, f"Initial ACN(xy) {initial_acn} is too low for a 5_1 knot"
    assert abs(initial_wr) > 0.1, f"Initial Writhe {initial_wr} is too low for T(2,5)"

    l_avg = knot.length / knot.num_nodes
    skipped_val = max(1, int(round(np.pi * knot.diameter / (2 * l_avg)))) if l_avg > 1e-9 else 3
    print(f"Using initial skipped={skipped_val}")

    # Parameters for the run
    # Run for a decent number of iterations, enable shifts, and try increasing delta
    # as described in the paper for Fig 5.
    num_iterations = 15000
    delta_start_iter = num_iterations // 3 # Start increasing delta after 1/3rd of iterations

    min_ld_achieved = run_sono_simplified(
        knot,
        max_iterations=num_iterations,
        num_of_it=100,
        scaling_factor=0.999,
        overlap_threshold=1e-5,
        delta=0.0001, # Start with very small delta as per paper (Fig 5 used 0.00001!)
        epsilon=0.1 * knot_diameter,
        skipped=skipped_val, # Pass the initial value, helper recalculates if needed
        enable_shift_nodes=True,
        shift_freq=10,
        enable_normalize=False, # Keep N constant for this specific test for now
        increase_delta_iter=delta_start_iter, # Iteration to increase delta
        delta_increase_factor=1000.0, # Increase delta significantly (paper went 0.00001 -> 0.1)
        delta_increase_duration=num_iterations // 3 # Keep delta high for 1/3rd
    )

    # Assertions
    final_acn = calculate_crossing_number_xy(knot)
    final_wr = calculate_writhe(knot)
    final_L_over_D = knot.length / knot.diameter if knot.diameter > 0 else float('inf')

    print(f"Final L/D={final_L_over_D:.3f} (Min achieved={min_ld_achieved:.3f}), ACN(xy)={final_acn:.1f}, Wr={final_wr:.3f}")

    assert final_acn >= 4.9, f"Final ACN(xy) {final_acn} is too low for a 5_1 knot"
    assert abs(final_wr) > 0.1, f"Final Writhe {final_wr} is too low for T(2,5)"

    # Check if L/D dropped below the symmetric threshold (24.2) towards the asymmetric (23.5)
    # We check the *minimum* value reached during the run, as the final state might relax slightly
    symmetric_threshold = 24.2
    target_asymmetric = 23.5
    assert min_ld_achieved < symmetric_threshold * 0.99, \
           f"Minimum L/D ({min_ld_achieved:.3f}) did not drop significantly below the symmetric threshold ({symmetric_threshold})"
    # Check if it got reasonably close to the target asymmetric value
    assert min_ld_achieved < target_asymmetric * 1.05, \
           f"Minimum L/D ({min_ld_achieved:.3f}) did not reach close to the target asymmetric value ({target_asymmetric})"

    # --- Save Outputs ---
    print(f"\nSaving final state for Symmetry Breaking test...")
    try:
        write_knot_to_file(knot, final_coords_filename)
        plot_knot(knot, title="Symmetry Breaking Test - Final State", save_path=final_plot_filename, show_plot=False)
    except Exception as e:
        pytest.fail(f"Error during output saving for Symmetry Breaking test: {e}")

    print("\nSymmetry Breaking Test Passed: Minimum L/D dropped below symmetric threshold.") 