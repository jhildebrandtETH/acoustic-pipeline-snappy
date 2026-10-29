"""Fresh-case parity with native Snappy, with the deliberate inlet exception."""
from pathlib import Path
import shutil
import tempfile
import unittest

from tools.preprocessing import prepare_case_directory
from tools.settings_report import read_dictionary, resolve_dictionary

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT.parent / 'CoreTemplates'


def read(case, name):
    return resolve_dictionary(read_dictionary(case / name, case, {}, []))


class NativeSolverDefaultsTests(unittest.TestCase):
    def test_prepared_cases_match_native_flow_settings_except_pressure_inlet(self):
        templates = sorted((ROOT / 'CoreTemplates').glob('*/*/*/system/controlDict'))
        self.assertEqual(len(templates), 9)
        with tempfile.TemporaryDirectory(prefix='ansys-native-defaults-') as folder:
            folder = Path(folder)
            for index, control in enumerate(templates):
                template = control.parent.parent
                relative = template.relative_to(ROOT / 'CoreTemplates')
                with self.subTest(template=str(relative)):
                    actual, native = folder / f'actual-{index}', folder / f'native-{index}'
                    prepare_case_directory(template, ROOT / 'Parameters', actual)
                    shutil.copytree(NATIVE / relative, native)
                    # Keep the current shared parameters, including resolved wall omega.
                    shutil.copytree(ROOT / 'Parameters', native / 'Parameters')
                    for name in ('system/fvSolution', 'system/fvSchemes'):
                        self.assertEqual(read(actual, name), read(native, name), name)
                    for source in sorted((native / '0').iterdir()):
                        if not source.is_file():
                            continue
                        name = '0/' + source.name
                        expected, configured = read(native, name), read(actual, name)
                        if source.name == 'p' and relative.parts[1] != 'DES':
                            inlet = configured['boundaryField']['inlet']
                            self.assertEqual(inlet['type'], 'totalPressure')
                            self.assertEqual(inlet['p0'], 'uniform 101325')
                            self.assertEqual(inlet['value'], 'uniform 101325')
                            expected['boundaryField']['inlet'] = inlet
                        self.assertEqual(configured, expected, name)
                    solution = read(actual, 'system/fvSolution')
                    self.assertEqual(solution['solvers']['p']['solver'], 'GAMG')
                    self.assertEqual(solution['solvers']['pFinal']['maxIter'], '50')
                    self.assertEqual(read(actual, 'system/fvSchemes')['wallDist']['method'],
                                     'meshWave')
                    if relative.parts[1] == 'kOmegaSST':
                        omega = read(actual, '0/omega')
                        self.assertEqual(omega['internalField'], 'uniform 0.0495')
                        self.assertEqual(omega['boundaryField']['inlet']['value'],
                                         'uniform 0.0495')
                        self.assertEqual(omega['boundaryField']['walls']['value'],
                                         'uniform 1000')

    def test_yplus_field_and_blade_area_monitor_remain_enabled(self):
        templates = sorted((ROOT / 'CoreTemplates').glob('*/*/*/system/controlDict'))
        with tempfile.TemporaryDirectory(prefix='ansys-yplus-defaults-') as folder:
            folder = Path(folder)
            for index, control in enumerate(templates):
                with self.subTest(template=str(control.relative_to(ROOT))):
                    case = folder / str(index)
                    prepare_case_directory(control.parent.parent, ROOT / 'Parameters', case)
                    functions = read(case, 'system/controlDict')['functions']
                    self.assertEqual(functions['yPlus']['enabled'], 'true')
                    self.assertEqual(functions['yPlus']['executeControl'], 'timeStep')
                    mean = functions['propellerYPlusMean']
                    self.assertEqual(mean['operation'], 'areaAverage')
                    self.assertEqual(mean['patch'], 'propeller')
                    self.assertEqual(mean['fields'], '(yPlus)')


if __name__ == '__main__':
    unittest.main()
