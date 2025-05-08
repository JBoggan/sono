import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
import os

# Use relative import
from .knot import Knot

def plot_knot(knot: Knot, title: str = "Knot Conformation", save_path: str = None, show_plot: bool = True):
    """
    Plots the 3D conformation of the knot.

    Args:
        knot: The Knot object to plot.
        title: The title for the plot.
        save_path: Optional path to save the plot image file.
                   The directory will be created if it doesn't exist.
        show_plot: Whether to display the plot window (plt.show()).
    """
    if knot.num_nodes < 2:
        print("Warning: Cannot plot knot with fewer than 2 nodes.")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get coordinates, including wrapping back to the start
    nodes = np.vstack([knot.nodes, knot.nodes[0]]) # Add first node at the end to close the loop
    x = nodes[:, 0]
    y = nodes[:, 1]
    z = nodes[:, 2]

    # Plot the knot backbone
    ax.plot(x, y, z, marker='o', markersize=2, linestyle='-', label=f'N={knot.num_nodes}')

    # Mark the starting node differently (optional)
    ax.plot(x[0:1], y[0:1], z[0:1], marker='x', markersize=8, color='red', linestyle='', label='Start Node')

    # --- Set plot appearance --- 
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Try to make aspect ratio equal - crucial for 3D perception
    # Calculate the range of data
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    ax.grid(True)

    # --- Save or Show --- 
    if save_path:
        try:
            output_dir = os.path.dirname(save_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory {output_dir}")
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig) # Close the figure if not showing to release memory


# Example usage (if run directly)
if __name__ == '__main__':
    if Knot is not None:
        print("Running visualization example...")
        # Create a simple trefoil
        t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        x = np.sin(t) + 2 * np.sin(2 * t)
        y = np.cos(t) - 2 * np.cos(2 * t)
        z = -np.sin(3 * t)
        coords_trefoil = np.stack([x, y, z], axis=-1)

        knot_trefoil = Knot(coordinates=coords_trefoil, diameter=0.5)

        plot_knot(knot_trefoil, title="Example Trefoil Knot (3_1)", save_path="./trefoil_plot.png")

        # Example of plotting without showing
        # plot_knot(knot_trefoil, title="Example Trefoil (No Show)", save_path="./trefoil_no_show.png", show_plot=False)
    else:
        print("Cannot run example because Knot class could not be imported.") 