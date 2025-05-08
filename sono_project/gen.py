import numpy as np

# Define triangle corners
p0 = np.array([0., 0., 0.])
p1 = np.array([1., 0., 0.])
p2 = np.array([0., 1., 0.])

# Calculate side lengths
l01 = np.linalg.norm(p1 - p0)
l12 = np.linalg.norm(p2 - p1)
l20 = np.linalg.norm(p0 - p2)
perimeter = l01 + l12 + l20

# Target number of points
n_total = 20

# Distribute points proportionally (approximately)
n01 = max(1, int(round(n_total * l01 / perimeter)))
n12 = max(1, int(round(n_total * l12 / perimeter)))
# Adjust last segment to ensure total is exactly n_total
n20 = n_total - n01 - n12
if n20 < 1:
    # Need at least one point per segment, readjust
    n01 = max(1, n01 - (1-n20)//2)
    n12 = max(1, n12 - (1-n20 + (1-n20)%2)//2)
    n20 = n_total - n01 - n12


print(f"Distributing {n_total} points: {n01} (P0->P1), {n12} (P1->P2), {n20} (P2->P0)")

# Generate points along each segment (excluding the endpoint)
# Use linspace to get evenly spaced points
points01 = np.linspace(p0, p1, num=n01, endpoint=False)
points12 = np.linspace(p1, p2, num=n12, endpoint=False)
points20 = np.linspace(p2, p0, num=n20, endpoint=False)

# Combine the points
triangle_coords_20 = np.vstack([points01, points12, points20])

print(f"Generated {triangle_coords_20.shape[0]} points.")

# Define the output filename
output_filename = "triangle_20_coords.txt"

# Save to file
try:
    np.savetxt(output_filename, triangle_coords_20, fmt='%.8f')
    print(f"Coordinates saved to {output_filename}")
except Exception as e:
    print(f"Error saving file: {e}")