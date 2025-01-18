#!/bin/bash
#SBATCH --job-name=linear_reg          
#SBATCH -p lrz-dgx-1-p100x8              
#SBATCH --gres=gpu:1                  
#SBATCH --time=800               
#SBATCH -o output_log_linear_transformation_updated.out
#SBATCH -e error_log_linear_transformation_updated.err   

enroot start -m code/ ssl << 'EOF'

cd ../code/representation-comparison

python linear_transformation.py --optimizer=adam --epochs=20 --batchsize=128

python linear_transformation.py --optimizer=adam --epochs=20 --batchsize=512

python linear_transformation.py --optimizer=adam --epochs=20 --batchsize=1024

EOF
