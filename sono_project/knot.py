import numpy as np

class Knot:
    """
    Represents a knot as a discrete sequence of points (nodes) in 3D space.
    """
    def __init__(self, coordinates: np.ndarray, diameter: float):
        """
        Initializes the Knot object.

        Args:
            coordinates: An Nx3 NumPy array of node coordinates.
            diameter: The diameter D of the rope/tube.
        """
        if not isinstance(coordinates, np.ndarray) or coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("Coordinates must be an Nx3 NumPy array.")
        if not isinstance(diameter, (float, int)) or diameter <= 0:
            raise ValueError("Diameter must be a positive number.")

        self.nodes = coordinates.astype(float) # Ensure float type
        self.diameter: float = float(diameter)
        self.num_nodes: int = self.nodes.shape[0]

        if self.num_nodes < 3:
             raise ValueError("A knot must have at least 3 nodes.")

        # Initial calculations
        self.length: float = self._calculate_total_length()
        self.target_leash_length: float = self.length / self.num_nodes

    def _calculate_total_length(self) -> float:
        """Calculates the total length L of the knot chain."""
        # Calculate segment lengths (including wrap-around distance between last and first node)
        segments = np.diff(self.nodes, axis=0, append=self.nodes[0:1])
        segment_lengths = np.linalg.norm(segments, axis=1)
        return np.sum(segment_lengths)

    def update_length(self) -> None:
        """Recalculates the knot's total length based on current node positions."""
        self.length = self._calculate_total_length()
        # Optionally recalculate target_leash_length if N hasn't changed,
        # but it's often updated during tightening or node normalization.
        # If only node positions changed, target_leash_length should ideally remain
        # based on the *initial* L/N or a desired value, not the current fluctuating L.
        # However, the paper isn't explicit here. We'll keep it simple for now.
        # If num_nodes > 0:
        #    self.target_leash_length = self.length / self.num_nodes

    def get_leash_lengths(self) -> np.ndarray:
        """Returns an array of current distances between adjacent nodes."""
        segments = np.diff(self.nodes, axis=0, append=self.nodes[0:1])
        return np.linalg.norm(segments, axis=1)

    def __repr__(self) -> str:
        return f"Knot(N={self.num_nodes}, D={self.diameter:.3f}, L={self.length:.3f}, l={self.target_leash_length:.4f})"

# Example usage (optional, for testing)
if __name__ == '__main__':
    # A simple triangle
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    knot = Knot(coordinates=coords, diameter=0.1)
    print(knot)
    print("Initial leash lengths:", knot.get_leash_lengths())

    # Simulate moving a node
    knot.nodes[1] = [1.5, 0, 0]
    knot.update_length() # Update total length
    print(knot)
    print("New leash lengths:", knot.get_leash_lengths())

    # Test invalid input
    try:
        Knot(np.array([[0,0],[1,1]]), 0.1)
    except ValueError as e:
        print("Caught expected error:", e)

    try:
        Knot(np.array([[0,0,0],[1,1,1]]), 0.1)
    except ValueError as e:
        print("Caught expected error:", e)

    try:
        Knot(coords, -0.1)
    except ValueError as e:
        print("Caught expected error:", e) 