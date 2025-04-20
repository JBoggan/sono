# SONO Knot Tightening Algorithm Recreation

## Project Overview

This project aims to recreate the **SONO (Shrink-On-No-Overlap)** algorithm for finding low-energy ("ideal") knot conformations, originally developed by the late Professor Piotr Pierański in the 1990s. Unfortunately, the original source code was lost after Professor Pierański's passing.

This implementation is based entirely on the descriptions, figures, and results presented in his paper:

> P. Pierański, "In search of ideal knots." In *Ideal Knots* (eds A. Stasiak, V. Katritch, L. H. Kauffman), pp. 20-41, World Scientific, Singapore, 1998. (An earlier version/draft might be the `.txt` file included in some contexts).

The goal is to create a functional Python implementation of the SONO algorithm and use the examples from the paper to build a test suite, verifying the recreation's behavior against the documented results.

## The SONO Algorithm

The SONO algorithm simulates the physical process of tightening a knot tied in a thick, flexible rope. It represents the knot as a closed chain of N discrete points (nodes) in 3D space, connected by inextensible "leashes" of length `l = L/N`. Each node is surrounded by a hard sphere of diameter `D`.

The core idea is to iteratively shrink the knot (by scaling down node coordinates) while preventing the spheres from overlapping.

The key procedures, as described in Section 3 of the paper, are:

1.  **`ControlLeashes (CL)`**: Adjusts the distances between adjacent nodes to maintain the target leash length `l`.
2.  **`FindNeighbours (FN)`**: Efficiently identifies pairs of nodes that are *not* adjacent along the chain but *are* close in 3D space, making them potential candidates for overlap.
3.  **`RemoveOverlaps (RO)`**: Detects actual overlaps (distance < `D`) between nodes identified by FN and pushes them apart symmetrically to a distance `D + delta`.
4.  **`ShiftNodes (SN)`**: Periodically shifts nodes along the knot's path by a small fraction, helping to smooth the conformation and prevent jamming.
5.  **Node Number Adjustment (`RNN`, `DNN`, `NNN`)**: Procedures to Reduce, Double, or Normalize the number of nodes `N`. Normalization typically aims for `N ≈ 10 * L / D`, where `L` is the current total length.
6.  **Tightening Step**: When overlaps are below a threshold, the knot's coordinates (and target leash length `l`) are scaled down by a factor `s < 1`, while the diameter `D` remains constant. This effectively shrinks or "tightens" the knot.

These procedures are orchestrated in a main loop (Section 3.5) that typically involves periodic neighbour finding, followed by iterations of CL, RO, optional SN, and the tightening step.

## Project Structure

```
sono_project/
├── main.py             # Main script to run the SONO simulation
├── knot.py             # Knot class definition
├── sono_procedures.py  # Implementation of CL, FN, RO, SN, NNN, etc.
├── knot_properties.py  # Calculation of ACN, Writhe, etc.
├── io_utils.py         # File reading/writing utilities
├── visualization.py    # (Optional) Plotting functions (Not Yet Implemented)
├── tests/                # Test suite based on paper examples
│   ├── test_untangling.py
│   ├── test_moffat.py
│   ├── test_perko_pair.py
│   └── test_symmetry_breaking.py
└── README.md           # This file
```

## Current Status & Implemented Components

*   **Core Algorithm**: All major procedures (CL, FN, RO, SN, RNN, DNN, NNN) described in Section 3 of the paper are implemented in `sono_procedures.py`.
*   **Knot Representation**: The `Knot` class in `knot.py` uses NumPy arrays to store node coordinates.
*   **Properties**: `knot_properties.py` includes functions to calculate the XY-plane crossing number (`calculate_crossing_number_xy`) and Writhe (`calculate_writhe`) using discrete approximations.
*   **I/O**: `io_utils.py` provides functions to read and write simple XYZ coordinate files.
*   **Main Executable**: `main.py` provides a command-line interface to run the SONO simulation with various parameters.
*   **Tests**: A test suite (`tests/`) using `pytest` is included, verifying behavior against key examples from the paper.

## Tests

The `tests/` directory contains tests designed to replicate the results shown in Professor Pierański's paper:

*   **`test_untangling.py`**: Based on Fig. 2 / Section 4.1. Tests the algorithm's ability to simplify a tangled unknot by removing empty loops.
*   **`test_moffat.py`**: Based on Fig. 3 / Section 4.2. Tests if the algorithm relaxes an initial T(3,2) trefoil knot configuration towards the expected lower-energy T(2,3) form.
*   **`test_perko_pair.py`**: Based on Fig. 4 / Section 4.3. Tests if the algorithm converges the two different initial configurations of the Perko pair (10_161 and 10_162) to the *same* final state (verified by L/D, ACN, Wr).
*   **`test_symmetry_breaking.py`**: Based on Fig. 5 / Section 5. Tests if the algorithm, potentially with parameter adjustments (like temporarily increasing `delta`), can break the initial symmetry of a T(2,5) knot (5_1) to find a lower L/D configuration, escaping a local minimum.

## How to Run

### Simulation

You can run the main SONO simulation using `main.py`. You need an input file with initial knot coordinates (Nx3, whitespace or comma-separated). The diameter (`-d`) is required.

```bash
# Example:
python3 sono_project/main.py path/to/your/knot_coords.txt -d 0.5 [OPTIONS]
```

Key options (see `python3 sono_project/main.py -h` for all):

*   `-o <output_file>`: Specify output file path.
*   `--max_iterations <N>`: Set total simulation iterations.
*   `-s <factor>`: Set scaling factor (e.g., 0.999).
*   `--delta <value>`: Set overlap removal gap (e.g., 0.005).
*   `--skipped <N>`: Set neighbour skip index (default calculated).
*   `--num_of_it <N>`: Set iterations between FindNeighbours calls (e.g., 200).

### Tests

Ensure you have `pytest` installed (`pip3 install pytest`). Run tests from the project root directory:

```bash
# Run all tests
python3 -m pytest tests/

# Run a specific test file (with output)
python3 -m pytest tests/test_perko_pair.py -s
```

## Future Work & Known Issues

*   **Performance**: The current `find_neighbours` implementation uses an O(N^2) check. This becomes a bottleneck for large numbers of nodes (`N > 1000`). Optimization using spatial partitioning (e.g., grid cells, k-d trees) would significantly improve performance.
*   **Test Reliability**: Some tests (e.g., symmetry breaking) are sensitive to simulation parameters (`delta`, `scaling_factor`, number of iterations, use of `shift_nodes`) and may require further tuning or algorithm optimization to pass consistently and efficiently.
*   **ACN Calculation**: The current `calculate_crossing_number_xy` only uses a single projection. A more robust Average Crossing Number calculation would average over multiple random projection directions.
*   **Visualization**: Implement `visualization.py` using a library like `matplotlib` to plot knot conformations.
*   **Test Refactoring**: Move shared helper functions (like `run_sono_simplified`) from individual test files into a common `tests/conftest.py`.
*   **Initial Coordinates**: Add more standard initial coordinate files for various knots to test against.