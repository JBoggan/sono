import argparse
import numpy as np
import time
import sys # Import sys for exit

# Import necessary components from our modules
from knot import Knot
from sono_procedures import (
    control_leashes,
    find_neighbours,
    remove_overlaps,
    shift_nodes,
    normalize_node_number # Add others like RNN/DNN later if controlled via args
)
# Import property calculation functions
from knot_properties import (
    calculate_crossing_number_xy,
    calculate_writhe
)
# Import the new I/O functions
from io_utils import read_knot_from_file, write_knot_to_file
# Import the visualization function
from visualization import plot_knot

# --- Remove Placeholder I/O --- 
# def read_knot_from_file(filepath: str) -> np.ndarray:
#     ...
# def write_knot_to_file(knot: Knot, filepath: str) -> None:
#     ...
# --- End Remove Placeholder I/O ---

def main():
    parser = argparse.ArgumentParser(description="Run the SONO knot tightening algorithm.")

    # Input/Output
    parser.add_argument("input_file", help="Path to the input file containing initial knot coordinates (Nx3 format).")
    parser.add_argument("-o", "--output_file", default="sono_output.txt", help="Path to save the final knot coordinates.")

    # Knot Parameters
    parser.add_argument("-d", "--diameter", type=float, required=True, help="Diameter (D) of the knot tube.")

    # SONO Algorithm Parameters (based on paper sections)
    parser.add_argument("--skipped", type=int, default=None, help="Index distance skip for neighbour finding (FN/RO). Default: round(pi*D / (2*l))")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Distance buffer for neighbour finding (FN).")
    parser.add_argument("--delta", type=float, default=0.001, help="Extra separation distance for overlap removal (RO). Should be > 0.")
    parser.add_argument("--shift_freq", type=int, default=10, help="Frequency (iterations) to run ShiftNodes (SN). 0 to disable.")
    parser.add_argument("--shift_frac", type=float, default=0.05, help="Fraction for ShiftNodes (SN).")
    parser.add_argument("--scaling_factor", "-s", type=float, default=0.999, help="Scaling factor (s < 1) for tightening step.")
    parser.add_argument("--overlap_threshold", type=float, default=1e-5, help="Average overlap threshold to trigger tightening.")
    parser.add_argument("--num_of_it", type=int, default=200, help="Iterations between FindNeighbours (FN) calls.")
    parser.add_argument("--max_iterations", type=int, default=50000, help="Total maximum iterations for the simulation.")
    parser.add_argument("--normalize_freq", type=int, default=10000, help="Frequency (iterations) to run NormalizeNodeNumber (NNN). 0 to disable.")
    parser.add_argument("--normalize_density", type=float, default=10.0, help="Target density for NormalizeNodeNumber (NNN).")
    parser.add_argument("--prop_calc_freq", type=int, default=100, help="Frequency (iterations) to calculate and print ACN/Wr.")

    # Visualization Parameters
    parser.add_argument("--visualize", action="store_true", help="Show a 3D plot of the final knot conformation.")
    parser.add_argument("--plot_save_path", type=str, default=None, help="Path to save the final knot plot image (e.g., final_knot.png).")

    args = parser.parse_args()

    # --- Initialization ---
    print("--- SONO Algorithm --- ")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Parameters: D={args.diameter}, s={args.scaling_factor}, delta={args.delta}, eps={args.epsilon}, num_it={args.num_of_it}, max_it={args.max_iterations}")

    # Use the imported read_knot_from_file function with error handling
    try:
        initial_coords = read_knot_from_file(args.input_file)
    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1) # Exit if input file is invalid

    # Create Knot object (add try-except for Knot initialization errors?)
    try:
        knot = Knot(coordinates=initial_coords, diameter=args.diameter)
    except ValueError as e:
        print(f"Error initializing knot: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Initial Knot State: {knot}")

    neighbours_list = []
    last_fn_time = time.time()
    start_time = time.time()

    # Calculate skipped dynamically if not provided
    use_dynamic_skipped = (args.skipped is None)
    current_skipped = args.skipped # Use provided value if given

    # --- Main Loop (Sec 3.5) ---
    print("\nStarting main SONO loop...")
    current_acn = 0.0
    current_wr = 0.0
    for iteration in range(args.max_iterations):
        # Recalculate skipped dynamically if needed, before finding neighbours
        if use_dynamic_skipped:
            if knot.target_leash_length > 1e-9:
                current_skipped = max(1, round(np.pi * knot.diameter / (2 * knot.target_leash_length)))
            else:
                current_skipped = knot.num_nodes // 2 # Fallback if leash length is zero

        # 2. Find Neighbours (periodically)
        if iteration % args.num_of_it == 0:
            fn_info = f"(skipped={current_skipped})"
            print(f"\nIter {iteration}: Finding neighbours {fn_info}...", end=" ")
            t_fn_start = time.time()
            # Use the potentially updated current_skipped value
            neighbours_list = find_neighbours(knot, current_skipped, args.epsilon)
            t_fn_end = time.time()
            num_neighbours = sum(len(nl) for nl in neighbours_list) // 2
            print(f"Found {num_neighbours} neighbour pairs in {t_fn_end - t_fn_start:.3f}s")
            last_fn_time = t_fn_end

        # 3. Control Leashes
        min_l, max_l = control_leashes(knot)

        # 4. Remove Overlaps
        max_ov, avg_ov = remove_overlaps(knot, neighbours_list, args.delta)

        # Optional: Shift Nodes (Sec 3.3)
        if args.shift_freq > 0 and iteration % args.shift_freq == 0:
             shift_nodes(knot, args.shift_frac)

        # 5. Tighten if overlaps are small
        if avg_ov < args.overlap_threshold:
            # Scale coordinates and leash length
            knot.nodes *= args.scaling_factor
            # Update length (L) and target leash length (l)
            knot.update_length()
            knot.target_leash_length *= args.scaling_factor # Keep l consistent with scaling
            # Diameter D remains unchanged

        # Optional: Normalize Node Number (Sec 3.4)
        if args.normalize_freq > 0 and iteration % args.normalize_freq == 0 and iteration > 0:
             print(f"\nIter {iteration}: Normalizing nodes (Target density={args.normalize_density})...")
             normalize_node_number(knot, target_density=args.normalize_density)
             # Knot object now has updated N, L, and target_leash_length

             # Recalculate skipped *based on the new leash length* if using dynamic skipping
             if use_dynamic_skipped:
                 if knot.target_leash_length > 1e-9:
                     current_skipped = max(1, round(np.pi * knot.diameter / (2 * knot.target_leash_length)))
                 else:
                     current_skipped = knot.num_nodes // 2 # Fallback
                 print(f"Iter {iteration}: Updated skipped to {current_skipped} after normalization.")

             # ALWAYS recalculate neighbours immediately after changing node count, using the updated skipped value
             fn_info = f"(skipped={current_skipped})"
             print(f"Iter {iteration}: Recalculating neighbours after normalization {fn_info} (N={knot.num_nodes})...")
             neighbours_list = find_neighbours(knot, current_skipped, args.epsilon)
             print(f"Iter {iteration}: Found {sum(len(nl) for nl in neighbours_list) // 2} neighbour pairs.")

        # Calculate and Print progress periodically
        if iteration % args.prop_calc_freq == 0 or iteration == args.max_iterations - 1:
            # Calculate properties
            current_acn = calculate_crossing_number_xy(knot)
            current_wr = calculate_writhe(knot)

            # Print status
            current_time = time.time()
            elapsed = current_time - start_time
            lambda_val = knot.length / knot.diameter if knot.diameter > 0 else float('inf')
            print(f"Iter {iteration:7d} [t={elapsed:.1f}s]: L/D={lambda_val:8.3f}, N={knot.num_nodes}, "
                  f"AvgOv={avg_ov:.2e}, ACN={current_acn:5.1f}, Wr={current_wr: 7.3f}, l_range=({min_l:.4f}-{max_l:.4f})")

    # --- Finalization ---
    end_time = time.time()
    print(f"\n--- Simulation Finished ({end_time - start_time:.2f}s) ---")

    # Final properties
    final_acn = calculate_crossing_number_xy(knot)
    final_wr = calculate_writhe(knot)

    print(f"Final Knot State: {knot}")
    lambda_final = knot.length / knot.diameter if knot.diameter > 0 else float('inf')
    print(f"Final L/D: {lambda_final:.4f}")
    print(f"Final ACN(xy): {final_acn:.1f}")
    print(f"Final Writhe: {final_wr:.4f}")

    # Save final coordinates
    try:
        write_knot_to_file(knot, args.output_file)
    except IOError as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

    # Plot final knot if requested
    if args.visualize or args.plot_save_path:
        try:
            plot_knot(
                knot,
                title=f"Final Conformation (L/D={lambda_final:.3f})",
                save_path=args.plot_save_path,
                show_plot=args.visualize # Only show if --visualize is set
            )
        except Exception as e:
            print(f"Error during plotting: {e}", file=sys.stderr)
            # Don't necessarily exit, simulation already finished

if __name__ == "__main__":
    main() 