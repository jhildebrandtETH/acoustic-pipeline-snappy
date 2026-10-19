#!/bin/bash
#SBATCH --job-name=acoustic-coordinator
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=acoustic-coordinator-%j.log
set -eo pipefail

# Adapt paths and add your usual environment activation or module commands.
cd /shared/acoustic-pipeline-snappy
/path/to/of_pipeline_env/bin/python main.py --sim-dir /shared/orders/propeller \
  --rpms 4000 5000 --mode AMI --turbulence kOmegaSST --wall-functions yes \
  --total-cores 32 --cores-per-case 8 --aerodynamics-only --end-on rev 2 \
  --slurm --slurm-config /shared/acoustic-pipeline-snappy/examples/slurm-config.json
