import os
import time 
import json
import itertools
import numpy as np
from datetime import datetime
from sklearn.neighbors import BallTree

def load_matrix(matrix_path):
    matrix = np.load(matrix_path)
    print(f"Loaded matrix with shape: {matrix.shape} from path: {matrix_path}")
    return matrix

def calculate_kNNs(matrix, k, folder_name):
    print("k-NN search starting ...")
    start_time = time.time()
    tree = BallTree(matrix, metric='euclidean', leaf_size=50)
    print("Tree created")
    indices = tree.query(matrix, k=k, return_distance=False)
    print("Shape of indices array is: ", indices.shape)
    end_time = time.time()
    print(f"Runtime for k-NN: {end_time - start_time:.6f} seconds\n")

    os.makedirs(folder_name, exist_ok=True)
    np.save(f"{folder_name}/knn_indices_k_{k}.npy", indices)
    return f"{folder_name}/knn_indices_k_{k}.npy"

def calculate_all_kNNs(list_matrix_paths, list_matrix_name_abbrs, k):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = os.path.join("results/kNN", timestamp)
    os.makedirs(folder_name, exist_ok=True)

    storage_paths = []
    for i in range(len(list_matrix_paths)):
        matrix = load_matrix(list_matrix_paths[i])
        storage_path = calculate_kNNs(matrix, k, folder_name+"/"+list_matrix_name_abbrs[i])
        storage_paths.append(storage_path)
    return storage_paths, folder_name, timestamp

def calculate_indices_overlap(indices_matrix1, indices_matrix2):
    overlaps = []
    print("k-NN overlap calculation starting ...")
    start_time = time.time()
    assert len(indices_matrix1) == len(indices_matrix2)
    for i in range(len(indices_matrix1)):
        set_indices1 = set(indices_matrix1[i])
        set_indices2 = set(indices_matrix2[i])
        intersection = set_indices1.intersection(set_indices2)
        overlaps.append(len(intersection))

    end_time = time.time()
    print(f"Runtime for k-NN overlap calculation: {end_time - start_time:.6f} seconds")
    overlaps =  np.array(overlaps)
    return np.sum(overlaps), np.mean(overlaps)

def store_knn_comparison_configs(matrix1_abbr, matrix2_abbr, matrix1_name, matrix2_name, matrix1_path, matrix2_path, overlaps_sum, overlaps_mean, k,timestamp, folder_name):
    results = {
        "timestamp": timestamp,
        "matrix1_name": matrix1_name,
        "matrix1_ckpt": matrix1_path,
        "matrix2_name": matrix2_name,
        "matrix2_ckpt": matrix2_path,
        "sum_overlaps": overlaps_sum.item(),
        "mean_overlaps": overlaps_mean.item(),
        "k": k,
        "self_included": True
    }
    with open(folder_name + f"/knn_comparison_{matrix1_abbr}-{matrix2_abbr}.json", 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Stored configuration under: {folder_name}/knn_comparison_{matrix1_abbr}-{matrix2_abbr}.json\n")

k = 5
matrix_name_abbrs = ["emb", "pair", "aug"]
matrix_names = ["Embedding matrix", "Eigenvector matrix - pair block", "Eigenvector matrix - augmentation group block"]
matrix_paths = ["/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/results/embeddings/curious-cosmos-122/chunks/embedding_1024_200.npy",
                "/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/matrices/generated/sparse_matrix_2_048_000/pair_block/eigenvectors_k_32.npy", 
                "/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/matrices/generated/sparse_matrix_2_048_000/aug_group_block/eigenvectors_k_32.npy"]

pairs = list(itertools.combinations(range(3), 2))
indices_paths, folder_name, timestamp = calculate_all_kNNs(matrix_paths, matrix_name_abbrs, k)
print("\n\n######################## Calculated all indices for all matrices ######################## \n\n")

for pair in pairs:
    overlaps_sum, overlaps_mean = calculate_indices_overlap(load_matrix(indices_paths[pair[0]]), load_matrix(indices_paths[pair[1]]))
    store_knn_comparison_configs(
        matrix_name_abbrs[pair[0]], matrix_name_abbrs[pair[1]],
        matrix_names[pair[0]], matrix_names[pair[1]],
        matrix_paths[pair[0]], matrix_paths[pair[1]],
        overlaps_sum=overlaps_sum, 
        overlaps_mean=overlaps_mean, 
        k=k,
        timestamp=timestamp, 
        folder_name=folder_name,
    )
   
    