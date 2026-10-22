"""Private sbatch entry point; does not acquire the coordinator's order lock."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.slurm import run_worker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim-dir', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run_worker(args.sim_dir, args.case, args.token) else 1)
