#!/bin/bash
#SBATCH --job-name=tsne        
#SBATCH -p lrz-cpu             
#SBATCH --qos=cpu                 
#SBATCH --time=800               
#SBATCH -o output_log_tsne.out
#SBATCH -e error_log_tsne.err   

enroot start -m code/ ssl << 'EOF'

date=$(date '+%Y-%m-%d %H:%M:%S')
echo $date

cd ../code/representation-comparison

python tsne.py --matrix_name="eigenvector matrix - pair block" --matrix_path="/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/matrices/generated/sparse_matrix_2_048_000/pair_block/eigenvectors_k_32.npy"
python tsne.py --tsne_metric="mahalanobis" --matrix_name="eigenvector matrix - pair block" --matrix_path="/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/matrices/generated/sparse_matrix_2_048_000/pair_block/eigenvectors_k_32.npy"
python tsne.py --tsne_metric="cosine" --matrix_name="eigenvector matrix - pair block" --matrix_path="/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/matrices/generated/sparse_matrix_2_048_000/pair_block/eigenvectors_k_32.npy"

python tsne.py --matrix_name="eigenvector matrix - augmentation group block" --matrix_path="/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/matrices/generated/sparse_matrix_2_048_000/aug_group_block/eigenvectors_k_32.npy"
python tsne.py --tsne_metric="mahalanobis" --matrix_name="eigenvector matrix - augmentation group block" --matrix_path="/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/matrices/generated/sparse_matrix_2_048_000/aug_group_block/eigenvectors_k_32.npy"
python tsne.py --tsne_metric="cosine" --matrix_name="eigenvector matrix - augmentation group block" --matrix_path="/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/matrices/generated/sparse_matrix_2_048_000/aug_group_block/eigenvectors_k_32.npy"

python tsne.py --matrix_name="embedding matrix - curious-cosmos-122" --matrix_path="/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/results/embeddings/curious-cosmos-122/chunks/embedding_1024_200.npy"
python tsne.py --tsne_metric="mahalanobis" --matrix_name="embedding matrix - curious-cosmos-122" --matrix_path="/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/results/embeddings/curious-cosmos-122/chunks/embedding_1024_200.npy"
python tsne.py --tsne_metric="cosine" --matrix_name="embedding matrix - curious-cosmos-122" --matrix_path="/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/results/embeddings/curious-cosmos-122/chunks/embedding_1024_200.npy"

EOF
