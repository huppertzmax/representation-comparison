#!/bin/bash
#SBATCH --job-name=dist_mes          
#SBATCH -p lrz-cpu             
#SBATCH --qos=cpu                 
#SBATCH --time=800               
#SBATCH -o output_log_distance_measures.out
#SBATCH -e error_log_distance_measures.err   

enroot create --name ssl2 code/container/ssl.sqsh
enroot start -m code/ ssl2 << 'EOF'

cd ../code/representation-comparison

python distance_measures_matrices_row_wise.py 

python distance_measures_matrices_row_wise.py --matrix2_name="Eigenvector matrix - pair block" --matrix2_path="/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/matrices/generated/sparse_matrix_2_048_000/pair_block/eigenvectors_k_32.npy"

python distance_measures_matrices.py 

python distance_measures_matrices.py --matrix2_name="Eigenvector matrix - pair block" --matrix2_path="/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/matrices/generated/sparse_matrix_2_048_000/pair_block/eigenvectors_k_32.npy"

EOF
