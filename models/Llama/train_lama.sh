#!/bin/bash
#SBATCH --job-name=llama_training        # Job name
#SBATCH --output=llama_training_%j.log   # Output file (where you can check the logs)
#SBATCH --error=llama_training_%j.err    # Error file
#SBATCH --partition=gpu                  # Partition to request (e.g., 'gpu', 'gpu-medium')
#SBATCH --gres=gpu:1                     # Request 1 GPU (you can change this if you need more GPUs)
#SBATCH --time=24:00:00                  # Time limit for the job (adjust as needed)
#SBATCH --mem=32G                        # Memory requirement (adjust as needed)
#SBATCH --cpus-per-task=8               # Number of CPU cores per task (adjust as needed)

# Load necessary modules
module load python/3.8.5                # Adjust Python version if needed
module load cuda/11.2                   # Adjust CUDA version if needed
module load cudnn/8.1                   # Adjust cuDNN version if needed

# Activate your Conda environment (if using Conda for Python dependencies)
# source activate your_conda_env  # Replace 'your_conda_env' with your Conda environment name

# Navigate to the directory where the script is located (optional if already in correct directory)
# cd /path/to/your/script/  # Change this path to where `instructiontuning.py` is located

# Run the Python training script
python instructiontuning.py  # This will run your instruction-tuning training script

