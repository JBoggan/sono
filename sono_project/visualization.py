import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
import os

# Use relative import
from .knot import Knot

def plot_knot(knot: Knot, title: str = "Knot Conformation", save_path: str = None, show_plot: bool = True, cmap_name: str = 'rainbow'):
    """
    Plots the 3D conformation of the knot, coloring segments by distance from origin.

    Args:
        knot: The Knot object to plot.
        title: The title for the plot.
        save_path: Optional path to save the plot image file.
                   The directory will be created if it doesn't exist.
        show_plot: Whether to display the plot window (plt.show()).
        cmap_name: Name of the matplotlib colormap to use (e.g., 'rainbow', 'viridis', 'plasma').
    """
    if knot.num_nodes < 2:
        print("Warning: Cannot plot knot with fewer than 2 nodes.")
        return

    fig = plt.figure(figsize=(9, 8)) # Slightly wider for colorbar
    ax = fig.add_subplot(111, projection='3d')

    # Get coordinates, including wrapping back to the start for segments
    nodes = knot.nodes
    n = knot.num_nodes

    # Calculate segment midpoints and their distances from origin
    midpoints = np.zeros((n, 3))
    distances = np.zeros(n)
    segments = []

    # Determine viewer point (max X, min Y, max Z of the knot)
    if n > 0:
        viewer_x = nodes[:, 0].max()
        viewer_y = nodes[:, 1].min()
        viewer_z = nodes[:, 2].max()
        viewer_point = np.array([viewer_x, viewer_y, viewer_z])
    else: # Should not happen due to early exit for n < 2
        viewer_point = np.array([0.0, 0.0, 0.0])

    for i in range(n):
        p1 = nodes[i]
        p2 = nodes[(i + 1) % n]
        segments.append((p1, p2))
        midpoints[i] = (p1 + p2) / 2.0
        distances[i] = np.linalg.norm(midpoints[i] - viewer_point)

    # Normalize distances for colormap
    min_dist = distances.min()
    max_dist = distances.max()
    if max_dist == min_dist:
        # Avoid division by zero if all points are equidistant (or only one segment)
        norm_distances = np.full(n, 0.5)
    else:
        norm_distances = (distances - min_dist) / (max_dist - min_dist)

    # Get the colormap
    try:
        cmap = cm.get_cmap(cmap_name)
    except ValueError:
        print(f"Warning: Colormap '{cmap_name}' not found. Using 'viridis'.")
        cmap = cm.get_cmap('viridis')

    # Plot each segment with its calculated color
    norm = mcolors.Normalize(vmin=min_dist, vmax=max_dist)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

    for i in range(n):
        p1, p2 = segments[i]
        color = cmap(norm_distances[i])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linestyle='-', marker='o', markersize=2)

    # Mark the starting node differently (optional)
    start_node = nodes[0]
    ax.plot([start_node[0]], [start_node[1]], [start_node[2]], marker='x', markersize=8, color='black', linestyle='', label='Start Node')

    # --- Set plot appearance --- 
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Try to make aspect ratio equal
    all_nodes_for_lim = np.vstack([nodes, nodes[0]]) # Use all nodes for limits
    x, y, z = all_nodes_for_lim[:, 0], all_nodes_for_lim[:, 1], all_nodes_for_lim[:, 2]
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    if max_range < 1e-6: max_range = 1.0 # Avoid zero range for collapsed knots
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add colorbar
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label('Distance from Viewer')

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

        # Plot with default rainbow
        plot_knot(knot_trefoil, title="Example Trefoil Knot (rainbow cmap)", save_path="./trefoil_plot_rainbow.png")
        # Plot with viridis
        # plot_knot(knot_trefoil, title="Example Trefoil Knot (viridis cmap)", save_path="./trefoil_plot_viridis.png", cmap_name='viridis', show_plot=False)

    else:
        print("Cannot run example because Knot class could not be imported.") 