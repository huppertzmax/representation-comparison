import os
import torch
import random
import json
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt

np.set_printoptions(precision=14)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = os.path.join("results/distances", timestamp)
os.makedirs(folder_name, exist_ok=True)

parser = ArgumentParser()

parser.add_argument("--normed_euc", type=bool, default=True)
parser.add_argument("--matrix1_name", type=str, default="Embedding matrix")
parser.add_argument("--matrix2_name", type=str, default="Eigenvector matrix")
parser.add_argument("--matrix1_path", type=str, default="/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/results/embeddings/curious-cosmos-122/chunks/embedding_1024_200.npy")
parser.add_argument("--matrix2_path", type=str, default="/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/matrices/generated/sparse_matrix_2_048_000/pair_block/eigenvectors_k_32.npy")
parser.add_argument("--num_samples_per_class", type=int, default=1024)
parser.add_argument("--num_augmentations", type=str, default=200)

args = parser.parse_args()

matrix1_name = args.matrix1_name
matrix2_name = args.matrix2_name
matrix1_path=  args.matrix1_path
matrix2_path = args.matrix2_path
use_normed_euc = args.normed_euc
elements_per_class = args.num_samples_per_class * args.num_augmentations

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using: {device} as device')

matrix1 = np.load(matrix1_path)
matrix1 = torch.from_numpy(matrix1).to(device)
matrix2 = np.load(matrix2_path)
matrix2 = torch.from_numpy(matrix2).to(device)

cos = torch.nn.CosineSimilarity(dim=-1)
euc = torch.nn.PairwiseDistance(p=2)
def normed_euc(row_m1, row_m2):
    normed_row_m1 = torch.nn.functional.normalize(row_m1, dim=-1)
    normed_row_m2 = torch.nn.functional.normalize(row_m2, dim=-1)
    return euc(normed_row_m1, normed_row_m2)

def compute_distances(matrix1, matrix2, elements_per_class):
    cos_dist = []
    euc_dist = []
    normed_euc_dist = []

    progress_bar = tqdm(range(matrix1.size(0)), leave=False)
    for row in progress_bar:
        row_m1 = matrix1[row, :]
        row_m2 = matrix2[row, :]
        
        c_dist = cos(row_m1, row_m2)
        c_dist = torch.sub(torch.ones_like(c_dist), c_dist)
        e_dist = euc(row_m1, row_m2)
        normed_e_dist = normed_euc(row_m1, row_m2)

        cos_dist.append(c_dist)
        euc_dist.append(e_dist)
        normed_euc_dist.append(normed_e_dist)

    cos_avg = sum(cos_dist) / len(cos_dist)
    euc_avg = sum(euc_dist) / len(euc_dist)
    normed_euc_avg = sum(normed_euc_dist) / len(normed_euc_dist)


    return cos_avg, euc_avg, normed_euc_avg

cos_dist, euc_dist, normed_euc_dist = compute_distances(matrix1, matrix2, elements_per_class)

results = {
    "timestamp": timestamp,
    "calculation_type": "row-wise",
    "matrix1_name": matrix1_name,
    "matrix1_ckpt": matrix1_path,
    "matrix2_name": matrix2_name,
    "matrix2_ckpt": matrix2_path,
    "cosine_distance": cos_dist,
    "normed_euclidean_distance": normed_euc_dist,
    "euclidean_distance": euc_dist,
}

with open(folder_name + "/distances.json", 'w') as f:
    json.dump(results, f, indent=4)