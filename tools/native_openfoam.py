"""Native single-node execution of the existing case command stream under Slurm."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from tools.common import emit_status


class NativeOpenFOAM:
    def __init__(self, case, cores):
        self.case = Path(case).resolve()
        self.cores = cores
        self.status = 'running'

    def reload(self):
        pass

    def stop(self):
        self.status = 'stopped'

    def remove(self, **kwargs):
        pass

    def command(self, cmd):
        tokens = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
        if len(tokens) != 3 or tokens[:2] != ['bash', '-c']:
            raise ValueError(f'Unsupported native OpenFOAM command: {tokens!r}')
        script = tokens[2]
        script = script.replace('source /opt/openfoam13/etc/bashrc', 'source "$ACOUSTIC_OPENFOAM_BASHRC"')
        script = script.replace('/simulation', '"${ACOUSTIC_CASE}"')
        launcher = json.loads(os.environ['ACOUSTIC_MPI_LAUNCHER'])
        launch = shlex.join([part.replace('{cores}', str(self.cores)) for part in launcher])
        script = re.sub(r'mpirun --allow-run-as-root --use-hwthread-cpus -np \d+', lambda _: launch, script)
        return ['bash', '-c', script]

    def run_native(self, cmd, description, live, callback):
        environment = dict(os.environ, ACOUSTIC_CASE=str(self.case), OMP_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        last_line = 'no command output'
        with subprocess.Popen(self.command(cmd), cwd=self.case, env=environment,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            for raw in process.stdout:
                line = raw.decode('utf-8', errors='replace').rstrip()
                if line:
                    last_line = line
                if live:
                    print(line, flush=True)
            code = process.wait()
        if code:
            message = f'{description} exited with code {code}: {last_line}'
            emit_status(callback, error=message, detail=message)
        return code == 0


def native_preflight(config):
    """Run on the allocated case node, after the site's environment setup."""
    script = '''source "$ACOUSTIC_OPENFOAM_BASHRC" || exit
test "$WM_PROJECT_VERSION" = 13 || { echo 'Foundation OpenFOAM 13 required'; exit 1; }
for tool in blockMesh surfaceFeatures snappyHexMesh splitBaffles decomposePar reconstructPar checkMesh createNonConformalCouples foamRun foamDictionary mapFields; do
    command -v "$tool" >/dev/null || exit 1
done
command -v "$ACOUSTIC_MPI_PROGRAM" >/dev/null
'''
    environment = dict(os.environ, ACOUSTIC_OPENFOAM_BASHRC=config['openfoam_bashrc'],
                       ACOUSTIC_MPI_PROGRAM=config['mpi_launcher'][0])
    subprocess.run(['bash', '-c', script], env=environment, check=True)
