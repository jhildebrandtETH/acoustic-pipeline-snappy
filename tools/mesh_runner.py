"""Run standard OpenFOAM 13 utilities natively or in an isolated container."""
import os
from pathlib import Path
import re
import subprocess

IMAGE = 'microfluidica/openfoam:13'


class Runner:
    def __init__(self, docker=False):
        self.docker = docker

    def run(self, case, program, flags, log):
        case = Path(case).resolve()
        if self.docker:
            command = ['docker', 'run', '--rm', '--network', 'none',
                       '--mount', f'type=bind,source={case},target=/case']
            if hasattr(os, 'getuid') and hasattr(os, 'getgid'):
                command += ['--user', f'{os.getuid()}:{os.getgid()}']
            command += ['--entrypoint', 'bash', IMAGE, '-lc',
                        'source /opt/openfoam13/etc/bashrc && exec "$@"',
                        'snappy-mesh', program, '-case', '/case', *flags]
        else:
            command = [program, '-case', str(case), *flags]
        with (case/log).open('w', encoding='utf-8') as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
        text = (case/log).read_text(encoding='utf-8', errors='replace')
        if result.returncode:
            raise RuntimeError(f'{program} failed; inspect {case/log}')
        if program == 'checkMesh':
            if 'Mesh OK.' not in text or re.search(r'Failed\s+\d+\s+mesh checks', text):
                raise RuntimeError(f'Mesh checks failed; inspect {case/log}')
            if not re.search(r'Number of regions:\s+1\b', text):
                raise RuntimeError(f'Expected one fluid region; inspect {case/log}')
