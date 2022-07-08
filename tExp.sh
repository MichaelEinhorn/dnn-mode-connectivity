#!/bin/bash
#SBATCH --job-name=p#
#SBATCH -t 0-18:00             		# Runtime in D-HH:MM
#SBATCH --gres=gpu:1 
#SBATCH --mem=32G
#SBATCH --output=outputLogs/p#.txt

nvidia-smi
cd dir
module load anaconda3/2021.05
conda activate torch
python train.py --dir=history/p# --dataset=CIFAR10 --data_path=CIFAR --model=arch# --epochs=200 --lr=0.1 --wd=w# --transform=trans# --num-workers 4 --parts=n# --offset=o# --save_freq=f# --momentum=0.9 --batch_size=128 --seed=s#
