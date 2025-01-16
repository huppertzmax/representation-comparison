import numpy as np
from scipy.sparse import load_npz, save_npz
from numpy.random import default_rng

part_size = 200
num_samples = 2
matrix_path = "/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/matrices/generated/sparse_matrix_2_048_000/aug_group_block/normalized_matrix_n_1024.npz"
storage_path = f"/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/matrices/generated/sparse_matrix_2_048_000/aug_group_block/normalized_matrix_n_1024_subset_{num_samples}.npz"

indices = []
rng = default_rng(123)
subset = rng.choice(part_size, size=num_samples, replace=False)
for i in range(10240):
    indices.extend([x+i*part_size for x in subset])

print("Start loading matrix: ")
matrix = load_npz(matrix_path)
print(matrix.shape)
print("Loaded matrix\n")
matrix = matrix[indices]
matrix.tocsr()
print(matrix.shape)

save_npz(storage_path, matrix)
print(matrix.shape)


