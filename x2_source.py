import numpy as np

from autoarray import numba_util
import autolens as al

pixels = 5
data = np.array([1.0, 2.0, 3.0])
noise_map = np.array([1.0, 1.0, 1.0])

mapping_matrix_0 = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]])

mapping_matrix_0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

mapping_matrix_1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])

curvature_matrix = al.util.leq.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=mapping_matrix_0, noise_map=noise_map
)

print()
print(curvature_matrix.shape)
print(curvature_matrix)

curvature_matrix = al.util.leq.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=mapping_matrix_1, noise_map=noise_map
)

print()
print(curvature_matrix.shape)
print(curvature_matrix)

mapping_matrix = np.hstack([mapping_matrix_0, mapping_matrix_1])

curvature_matrix = al.util.leq.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=mapping_matrix, noise_map=noise_map
)

print()
print(curvature_matrix.shape)
print(curvature_matrix)
