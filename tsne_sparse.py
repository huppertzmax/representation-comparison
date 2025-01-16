import os
from datetime import datetime
import numpy as np
from scipy.sparse import load_npz
from tsne import run_tsne, visualize_tsne, store_config

num_parts = 10
num_samples = 2*1024
matrix_path = f"/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/matrices/generated/sparse_matrix_2_048_000/aug_group_block/normalized_matrix_n_1024_subset_2.npz"
matrix_name = "population augmentation graph - augmentation group matrix"

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = os.path.join("results/tsne", timestamp)
os.makedirs(folder_name, exist_ok=True)

labels = np.array([i for i in range(num_parts) for _ in range(num_samples)])

print("Start loading matrix")
matrix = load_npz(matrix_path)
matrix = matrix.toarray()
print("Finished loading matrix\n")

tsne_results = run_tsne(matrix, 300, "euclidean", folder_name)
del matrix

visualize_tsne(tsne_results, labels, matrix_name, folder_name)
store_config(timestamp, matrix_name, matrix_path, num_parts, num_samples, 204800, 300, "euclidean", folder_name)
