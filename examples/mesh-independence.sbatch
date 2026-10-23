#!/bin/bash
#SBATCH --job-name=mesh-independence-coordinator
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --output=mesh-independence-coordinator-%j.log
#SBATCH --no-requeue

# Submit from the NORMAL snappy repository root with of_pipeline_env active:
#   sbatch examples/mesh-independence.sbatch
# Add your site's coordinator partition/account using sbatch -p ... -A ... .
# The header allocates ONLY the coordinator, not the case CPU budget below.
# Set its wall time for the entire study plus case queue waits; adjust to site limits.
# Do not add an array, MPI launch, or a multi-task srun around this coordinator.
set -euo pipefail

# Environment overrides can also be exported before sbatch.
PIPELINE_DIR="${PIPELINE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
SIM_DIR="${SIM_DIR:-$HOME/run/mesh_independence}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SLURM_CONFIG="${SLURM_CONFIG:-$SIM_DIR/slurm.json}"

RPM="${RPM:-4000}"
MODE="${MODE:-AMI}"
TURBULENCE="${TURBULENCE:-kOmegaSST}"
WALL_FUNCTIONS="${WALL_FUNCTIONS:-yes}"
REVOLUTIONS="${REVOLUTIONS:-30}"
TOTAL_CORES="${TOTAL_CORES:-150}"
CORES_PER_CASE="${CORES_PER_CASE:-50}"
# on: full acoustic postprocessing, including the configured 5000 Hz plot cap.
# off: aerodynamic mesh-independence study only.
ACOUSTICS="${ACOUSTICS:-on}"

cd "$PIPELINE_DIR"
if [[ ! -f main.py || ! -f Parameters/blockMeshDict.cpp || -f tools/ansys_mesh.py ]]; then
    echo 'PIPELINE_DIR must point to the normal snappy pipeline root, not ansys_pipeline.' >&2
    exit 1
fi

# The pipeline restores saved settings and reattaches existing case jobs on resume.
# Resubmitting does not rebuild completed cases or create a second study.
if [[ -f "$SIM_DIR/simulation_order.json" ]]; then
    exec "$PYTHON_BIN" -u main.py --sim-dir "$SIM_DIR" --slurm --resume
fi

if [[ ! -f "$SLURM_CONFIG" ]]; then
    echo "Missing case-job Slurm config: $SLURM_CONFIG" >&2
    exit 1
fi

# Supply exactly one propeller in ~/run/mesh_independence/STL/ before submitting.
# The same geometry, RPM, physical settings and run length are used at every size.
"$PYTHON_BIN" - "$SIM_DIR" <<'PY'
import sys
from pathlib import Path
from tools.openfoam import find_source_stls
inputs = find_source_stls(Path(sys.argv[1]) / 'STL')
if len(inputs) != 1:
    raise SystemExit('Mesh independence requires exactly one propeller STL in the order STL/ directory.')
PY

# --study-values uses literal "..." separators, not range syntax.
# Produce (4 12 4)...(5 15 5)... ... ...(18 54 18): 15 independent cases.
STUDY_VALUES=""
for ((n=4; n<=18; n++)); do
    [[ -z "$STUDY_VALUES" ]] || STUDY_VALUES+='...'
    STUDY_VALUES+="($n $((3*n)) $n)"
done

acoustic_args=()
case "$ACOUSTICS" in
    on) acoustic_args=(--acoustic-surface impermeable) ;;
    off) acoustic_args=(--aerodynamics-only) ;;
    *) echo 'ACOUSTICS must be on or off.' >&2; exit 1 ;;
esac

printf 'Starting 15 mesh levels at %s RPM in %s\n' "$RPM" "$SIM_DIR"
printf 'blocks_resolution: %s\n' "$STUDY_VALUES"
exec "$PYTHON_BIN" -u main.py \
    --sim-dir "$SIM_DIR" \
    --rpms "$RPM" --mode "$MODE" --turbulence "$TURBULENCE" \
    --wall-functions "$WALL_FUNCTIONS" --field-init off \
    --total-cores "$TOTAL_CORES" --cores-per-case "$CORES_PER_CASE" \
    --study --study-file blockMeshDict.cpp \
    --study-parameter blocks_resolution --study-values "$STUDY_VALUES" \
    --end-on rev "$REVOLUTIONS" \
    "${acoustic_args[@]}" \
    --slurm --slurm-config "$SLURM_CONFIG"
