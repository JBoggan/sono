import numpy as np
import pytest
import os
import sys

# Add project root to sys.path to allow importing from sono_project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from sono_project.knot import Knot
from sono_project.sono_procedures import (
    control_leashes,
    find_neighbours,
    remove_overlaps,
    shift_nodes
)
from sono_project.knot_properties import (
    calculate_crossing_number_xy,
    calculate_writhe
)
# Import I/O and visualization
from sono_project.io_utils import write_knot_to_file
from sono_project.visualization import plot_knot


def run_sono_simplified(
    knot: Knot,
    max_iterations: int,
    num_of_it: int = 50, # Frequency for FindNeighbours
    scaling_factor: float = 0.998,
    overlap_threshold: float = 1e-4,
    delta: float = 0.01,
    epsilon: float = 0.1, # Needs to be > 0
    skipped: int = 1
    ) -> None:
    """Runs a simplified SONO loop for testing purposes."""
    neighbours_list = []
    print(f"\nRunning simplified SONO for {max_iterations} iterations...")
    print(f"Initial state: {knot}, L/D={knot.length/knot.diameter:.2f}")

    for iteration in range(max_iterations):
        if iteration % num_of_it == 0:
            neighbours_list = find_neighbours(knot, skipped, epsilon)

        min_l, max_l = control_leashes(knot)
        max_ov, avg_ov = remove_overlaps(knot, neighbours_list, delta)
        # shift_nodes(knot, 0.05) # Optional: Add shift nodes if needed

        if avg_ov < overlap_threshold:
            knot.nodes *= scaling_factor
            knot.update_length()
            knot.target_leash_length *= scaling_factor

        if iteration % (max_iterations // 10) == 0:
             lambda_val = knot.length / knot.diameter if knot.diameter > 0 else float('inf') # Add check
             print(f"Iter {iteration}: L/D={lambda_val:.3f}, AvgOv={avg_ov:.2e}")

    final_lambda = knot.length / knot.diameter if knot.diameter > 0 else float('inf') # Add check
    print(f"Final state: {knot}, L/D={final_lambda:.3f}")


def test_untangle_simple_loop():
    """
    Test if SONO can remove a simple Reidemeister-I type loop from an unknot.
    Start with a circle that has a small twist/loop added.
    Expect it to relax to a near-perfect circle with ACN/Wr near 0.
    """
    # --- Test Setup ---
    output_dir = os.path.join(project_root, "test_outputs")
    initial_coords_filename = os.path.join(output_dir, "untangle_initial_coords.txt")
    initial_plot_filename = os.path.join(output_dir, "untangle_initial_plot.png")
    final_coords_filename = os.path.join(output_dir, "untangle_final_coords.txt")
    final_plot_filename = os.path.join(output_dir, "untangle_final_plot.png")
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists

    # Create a near-circle with a small loop
    n_points = 50
    radius = 5.0
    loop_radius = 0.5
    loop_offset = radius - loop_radius * 1.5
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Base circle
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros(n_points)

    # Add a small loop in the xy plane around theta = 0
    loop_indices = np.where((theta < 0.5) | (theta > 2*np.pi - 0.5))[0]
    loop_theta_local = np.linspace(-np.pi, np.pi, len(loop_indices))
    x[loop_indices] = loop_offset + loop_radius * np.cos(loop_theta_local + np.pi)
    y[loop_indices] = loop_radius * np.sin(loop_theta_local + np.pi)
    # Add a small z-component to make it 3D and potentially have non-zero writhe initially
    z[loop_indices] = 0.1 * np.sin(loop_theta_local * 0.5)

    initial_coords = np.stack([x, y, z], axis=-1)
    knot_diameter = 1.0

    knot = Knot(coordinates=initial_coords, diameter=knot_diameter)

    # --- Save Initial State --- 
    print(f"\nSaving initial state for untangling test...")
    try:
        write_knot_to_file(knot, initial_coords_filename)
        plot_knot(knot, title="Untangling Test - Initial State", save_path=initial_plot_filename, show_plot=False)
    except Exception as e:
        pytest.fail(f"Error during initial output saving for untangling test: {e}")

    initial_acn = calculate_crossing_number_xy(knot)
    initial_wr = calculate_writhe(knot)
    initial_L = knot.length
    print(f"\nUntangling Test: Initial N={knot.num_nodes}, D={knot.diameter}")
    print(f"Initial L={initial_L:.3f}, ACN(xy)={initial_acn:.1f}, Wr={initial_wr:.3f}")

    # Determine initial 'skipped' based on average leash length
    l_avg = knot.length / knot.num_nodes
    skipped_val = max(1, round(np.pi * knot.diameter / (2 * l_avg))) if l_avg > 1e-9 else 3
    print(f"Using skipped={skipped_val}")

    # Run the simplified SONO process
    run_sono_simplified(
        knot,
        max_iterations=2000,
        num_of_it=50,
        scaling_factor=0.998,
        overlap_threshold=1e-4,
        delta=0.01,
        epsilon=0.1 * knot_diameter, # Scale epsilon with diameter
        skipped=skipped_val
    )

    # Assertions: Check if the knot simplified
    final_acn = calculate_crossing_number_xy(knot)
    final_wr = calculate_writhe(knot)
    final_L = knot.length

    print(f"Final L={final_L:.3f}, ACN(xy)={final_acn:.1f}, Wr={final_wr:.3f}")

    # Check if length decreased significantly
    assert final_L < initial_L * 0.9, f"Length did not decrease significantly ({initial_L:.3f} -> {final_L:.3f})"

    # Check if crossings and writhe reduced towards zero
    # Allow some tolerance for numerical precision and discrete approximation
    assert np.isclose(final_acn, 0.0, atol=0.1), f"Final ACN(xy) {final_acn} is not close to 0"
    assert np.isclose(final_wr, 0.0, atol=0.1), f"Final Writhe {final_wr} is not close to 0"

    # Optional: Check if it resembles a circle (e.g., low variance in distance from center)
    center = np.mean(knot.nodes, axis=0)
    distances = np.linalg.norm(knot.nodes - center, axis=1)
    radius_variance = np.var(distances)
    print(f"Variance of node distance from center: {radius_variance:.4f}")
    # We expect low variance for a circle, but hard to set a strict threshold
    final_radius_approx = final_L / (2 * np.pi) if final_L > 0 else 0.1
    assert radius_variance < 0.1 * final_radius_approx**2, "Knot does not resemble a circle (high radius variance)"

    # --- Save Outputs ---
    print(f"\nSaving final state for untangling test...")
    try:
        write_knot_to_file(knot, final_coords_filename)
        plot_knot(knot, title="Untangling Test - Final State", save_path=final_plot_filename, show_plot=False)
    except Exception as e:
        pytest.fail(f"Error during output saving for untangling test: {e}") 