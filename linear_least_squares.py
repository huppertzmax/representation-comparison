import os
import json
from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import cupy as cp 


parser = ArgumentParser()

parser.add_argument("--matrix1_name", type=str, default="Embedding matrix")
parser.add_argument("--matrix2_name", type=str, default="Eigenvector matrix - augmentation group block")
parser.add_argument("--matrix1_path", type=str, default="/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/results/embeddings/curious-cosmos-122/chunks/embedding_1024_200.npy")
parser.add_argument("--matrix2_path", type=str, default="/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/matrices/generated/sparse_matrix_2_048_000/aug_group_block/eigenvectors_k_32.npy")

args = parser.parse_args()

matrix1_name = args.matrix1_name
matrix2_name = args.matrix2_name
matrix1_path=  args.matrix1_path
matrix2_path = args.matrix2_path

A = np.load(matrix1_path)
B = np.load(matrix2_path)

A = cp.asarray(A)
B = cp.asarray(B)
print(f"Loaded matrix A: {matrix1_name} with shape {A.shape}")
print(f"Loaded matrix B: {matrix2_name} with shape {B.shape}\n")

AT = A.T
ATA = AT @ A
ATB = AT @ B

try:
    ATA_inv = cp.linalg.inv(ATA)
except cp.linalg.LinAlgError:
    print("Calculating the pseudo inverse of ATA")
    ATA_inv = cp.linalg.pinv(ATA)

inverse_correct = cp.allclose(ATA @ ATA_inv, cp.eye(ATA.shape[0]), atol=1e-03)
print(f"Inverse is correct: {inverse_correct}")

T = ATA_inv @ ATB
print(f"Calculated transformation matrix T with shape {T.shape}")

B_pred = A @ T
print(f"Calculated prediction of matrix B with shape {B_pred.shape}\n")
error = cp.linalg.norm(B - B_pred) 
norm_B = cp.linalg.norm(B)
normed_error = error / norm_B
print(f"Error between B and prediction of B (Frobenius norm): {error}")
print(f"Normed error between B and prediction of B divided by norm of B: {normed_error}")

T = cp.asnumpy(T)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = os.path.join("results/linear-transformation", f"{timestamp}-matrices")
os.makedirs(folder_name, exist_ok=True)
save_path = os.path.join(folder_name, "transformation_matrix.npy")
np.save(save_path, T)
print(f"\nStored transformation matrix T under: {save_path}")

results = {
    "timestamp": timestamp,
    "matrix1_name": matrix1_name,
    "matrix2_name": matrix2_name,
    "matrix1_path": matrix1_path,
    "matrix2_path": matrix2_path,
    "frobenius_norm": error.item(),
    "frobenius_norm_relativ": normed_error.item(),
}

with open(folder_name + "/config.json", 'w') as f:
    json.dump(results, f, indent=4)