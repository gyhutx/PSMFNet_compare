#!/bin/bash
#SBATCH --job-name=vmunetv2_train                                      
#SBATCH -N 1
#SBATCH -c 4                                                   
#SBATCH --gres=gpu:1                        
#SBATCH -e %j.err                                       
#SBATCH -o %j.log                                       
#SBATCH -p GPUFEE08
#SBATCH --constraint="Python"                                   
source /gpfs/home/WB23301078/anaconda3/bin/activate
conda activate tmamba
srun python /gpfs/home/WB23301078/mamba/vmunetv2/VM-UNetV2-main/train_isic_all.py
