#!/bin/sh
#SBATCH -c 1                # Request 1 CPU core
#SBATCH -t 0-02:00          # Runtime in D-HH:MM, minimum of 10 mins
#SBATCH --partition=gpmoo-b  # Partition to submit to
 
#SBATCH --mem=10G           # Request 10G of memory
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written
#SBATCH --gres=gpu:1        # Request one GPUs           

# Command you want to run on the cluster
# Notice, you must set-up testEval correctly as a conda virtual environment
# Calling this full path makes sure you are running the correct package versions
python instructiontuning.py
