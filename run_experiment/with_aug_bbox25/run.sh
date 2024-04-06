#! /usr/bin/bash

#SBATCH --job-name=model
#SBATCH --output=out.%x.o%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --partition=pbatch
#SBATCH --time=12:00:00

srun /g/g11/eisenbnt/venvs/base/bin/python3 \
    -u /g/g11/eisenbnt/projects/flir_yolov5/run_experiment/with_augmentations/__main__.py
