import sys
from fileio import load_volume_mesh
import numpy as np

filepath = sys.argv[1]
mesh = load_volume_mesh(filepath)

vertices = [ [p[0], p[1], p[2]] for p in mesh.ngmesh.Points() ]
vertices = np.array(vertices)
x = vertices[:, 0]
y = vertices[:, 1]
z = vertices[:, 2]

x = x*0.001
y = y*0.001
z = z*0.001

print("x", min(x), max(x))
print("y", min(y), max(y))
print("z", min(z), max(z))
