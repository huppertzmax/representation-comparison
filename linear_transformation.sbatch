#!/bin/bash
#SBATCH --job-name=train_job          
#SBATCH -p lrz-dgx-1-p100x8              
#SBATCH --gres=gpu:1                  
#SBATCH --time=500               
#SBATCH -o output_log_linear_transformation.out
#SBATCH -e error_log_linear_transformation.err   

enroot start -m code/ ssl << 'EOF'

cd ../code/representation-comparison

python linear_transformation.py --optimizer=sgd --epochs=50 --batchsize=128

python linear_transformation.py --optimizer=adam --epochs=50 --batchsize=128

EOF
