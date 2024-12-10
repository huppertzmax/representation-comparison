#!/bin/bash
#SBATCH --job-name=train_job          
#SBATCH -p lrz-dgx-1-p100x8              
#SBATCH --gres=gpu:1                  
#SBATCH --time=300               
#SBATCH -o output_log_linear_transformation.out
#SBATCH -e error_log_linear_transformation.err   

enroot create --name ssl_container1 ./code/container/kernel-infonce.sqsh
enroot start -m code/ ssl_container1 << 'EOF'

pip install ipdb

cd ../code/representation-comparison

python linear_transformation.py

EOF
