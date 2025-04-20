import numpy as np
import os

from knot import Knot # Assuming knot.py is in the same directory or accessible

def read_knot_from_file(filepath: str) -> np.ndarray:
    """
    Reads knot coordinates from a text file.

    Expects a file with N rows and 3 columns (x, y, z), separated by whitespace
    or commas. Lines starting with '#' are ignored.

    Args:
        filepath: Path to the input file.

    Returns:
        An Nx3 NumPy array of node coordinates.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    try:
        # Attempt loading with default whitespace delimiter
        coords = np.loadtxt(filepath, comments='#')
    except ValueError:
        try:
            # If that fails, try with comma delimiter
            coords = np.loadtxt(filepath, comments='#', delimiter=',')
        except ValueError as e_comma:
            raise ValueError(f"Failed to parse file {filepath}. Ensure it has 3 columns (x y z) separated by whitespace or commas. Error: {e_comma}")
    except Exception as e:
        raise IOError(f"An unexpected error occurred while reading {filepath}: {e}")

    if coords.ndim == 1 and coords.shape[0] == 3:
        # Handle case where loadtxt returns a 1D array for a single node
        coords = coords.reshape(1, 3)
    elif coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Invalid coordinate format in file {filepath}. Expected Nx3 shape, got {coords.shape}.")

    print(f"Successfully read {coords.shape[0]} nodes from {filepath}")
    return coords.astype(float) # Ensure float type

def write_knot_to_file(knot: Knot, filepath: str, fmt: str = '%.8f', delimiter: str = ' ') -> None:
    """
    Writes the knot coordinates to a text file.

    Args:
        knot: The Knot object containing the coordinates to write.
        filepath: Path to the output file.
        fmt: Format string for NumPy savetxt.
        delimiter: Delimiter for NumPy savetxt.

    Raises:
        IOError: If writing to the file fails.
    """
    print(f"Writing {knot.num_nodes} nodes to {filepath}...", end=" ")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory {output_dir}", end="... ")

        np.savetxt(filepath, knot.nodes, fmt=fmt, delimiter=delimiter)
        print("Save complete.")
    except Exception as e:
        raise IOError(f"Error writing knot data to file {filepath}: {e}")

# Example usage
if __name__ == '__main__':
    # Create a dummy knot
    test_coords = np.array([[0,0,0],[1,1,1],[2,0,0]])
    test_knot = Knot(test_coords, 0.1)

    # Test writing
    test_write_path = "./temp_knot_output.txt"
    try:
        write_knot_to_file(test_knot, test_write_path)
        print(f"Check contents of {test_write_path}")

        # Test reading
        read_coords = read_knot_from_file(test_write_path)
        print("Read coordinates:\n", read_coords)
        assert np.allclose(test_coords, read_coords)
        print("Read coordinates match written coordinates.")

    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"Error during test: {e}")
    finally:
        # Clean up the dummy file
        if os.path.exists(test_write_path):
            os.remove(test_write_path)
            print(f"Removed temporary file {test_write_path}")

    # Test reading a non-existent file
    try:
        read_knot_from_file("non_existent_file.txt")
    except FileNotFoundError as e:
        print(f"Caught expected error: {e}")

    # Test reading an invalid file
    invalid_file_path = "./temp_invalid_knot.txt"
    with open(invalid_file_path, 'w') as f:
        f.write("1 2\n3 4 5 6")
    try:
        read_knot_from_file(invalid_file_path)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    finally:
        if os.path.exists(invalid_file_path):
            os.remove(invalid_file_path) 