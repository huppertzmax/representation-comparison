#!/bin/bash
#SBATCH --job-name=kNN          
#SBATCH -p lrz-cpu             
#SBATCH --qos=cpu                 
#SBATCH --time=800               
#SBATCH -o output_log_knn.out
#SBATCH -e error_log_knn.err   

enroot create --name ssl2 code/container/ssl.sqsh
enroot start -m code/ ssl2 << 'EOF'

cd ../code/representation-comparison

python knn.py

EOF
