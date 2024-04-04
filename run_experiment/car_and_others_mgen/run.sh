#! /usr/bin/bash

#SBATCH --job-name=model
#SBATCH --output=out.%x.o%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --partition=pvis

srun /g/g11/eisenbnt/venvs/base/bin/python3 \
	        -u /g/g11/eisenbnt/projects/flir_yolov5/run_experiment/car_and_others_mgen/__main__.py
