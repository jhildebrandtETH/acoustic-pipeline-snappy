"""Keep a single environment update sufficient for the bundled workflows."""
import ast
import os
from pathlib import Path
import re
import sys
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


def pip_requirements(path):
    """Read the pip list without requiring another YAML dependency for tests."""
    text = path.read_text(encoding='utf-8').split('  - pip:', 1)[1]
    return [line.strip()[2:] for line in text.splitlines()
            if line.strip().startswith('- ')]


def package_name(requirement):
    match = re.match(r'[A-Za-z0-9][A-Za-z0-9_.-]*', requirement)
    return match[0].lower().replace('_', '-') if match else None


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.requirements = pip_requirements(ROOT / 'of_pipeline_env.yml')
        self.packages = {package_name(item) for item in self.requirements}

    def test_runtime_and_test_imports_are_declared(self):
        local = {p.stem for p in ROOT.glob('*.py')}
        local |= {'tools', 'tests', 'acousticSolver', 'foamacoustics', 'Scripts', 'mesh_case_fixture'}
        aliases = {'vtkmodules': 'vtk', 'mpl_toolkits': 'matplotlib'}
        missing = {}
        for directory, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in {
                'output', 'runs', '__pycache__', '.git', '.venv', '.ipynb_checkpoints'}]
            for filename in files:
                if not filename.endswith('.py'):
                    continue
                path = Path(directory) / filename
                tree = ast.parse(path.read_text(encoding='utf-8-sig'))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = [item.name for item in node.names]
                    elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                        names = [node.module]
                    else:
                        continue
                    for name in names:
                        module = name.split('.')[0]
                        if module in local or module in sys.stdlib_module_names:
                            continue
                        package = aliases.get(module, module).lower().replace('_', '-')
                        if package not in self.packages:
                            missing.setdefault(package, set()).add(str(path.relative_to(ROOT)))
        self.assertFalse(missing, f'Add these imports to of_pipeline_env.yml: {missing}')

    def test_acoustic_notebook_and_pdf_test_dependencies_are_declared(self):
        project = tomllib.loads((ROOT / 'acousticSolver/pyproject.toml').read_text(encoding='utf-8'))['project']
        required = project['dependencies'] + project['optional-dependencies']['notebook']
        required += (ROOT / 'requirements-test.txt').read_text(encoding='utf-8').splitlines()
        for item in required:
            with self.subTest(requirement=item):
                self.assertIn(package_name(item), self.packages)
        self.assertIn('-e ./acousticSolver', self.requirements)

    def test_repository_and_ansys_environments_have_the_same_dependencies(self):
        parent_environment = ROOT.parent / 'of_pipeline_env.yml'
        if not parent_environment.exists():
            self.skipTest('standalone ANSYS pipeline copy')
        self.assertEqual(pip_requirements(parent_environment), self.requirements)
