#!/bin/bash
#SBATCH --job-name=mesh-independence
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --output=mesh-independence-%j.log
#SBATCH --error=mesh-independence-%j.err

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate of_pipeline_env

cd /home/jhildebrandt/projects/acoustic-pipeline-snappy

study_values=""
for n in {4..18}; do
    study_values+="($n $((3*n)) $n)..."
done
study_values="${study_values%...}"

python -u main.py \
    --sim-dir "$HOME/run/mesh_independence" \
    --rpms 4000 \
    --mode AMI \
    --turbulence kOmegaSST \
    --wall-functions yes \
    --field-init off \
    --total-cores 150 \
    --cores-per-case 30 \
    --study \
    --study-file blockMeshDict.cpp \
    --study-parameter blocks_resolution \
    --study-values "$study_values" \
    --end-on rev 30 \
    --acoustic-surface impermeable \
    --slurm \
    --slurm-config "$HOME/run/mesh_independence/slurm.json"
