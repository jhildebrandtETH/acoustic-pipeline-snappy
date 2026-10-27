import json
from pathlib import Path
import tempfile
import unittest
from tools import ansys_mesh as am


class SeparateHubTests(unittest.TestCase):
    def test_explicit_mapping_preserves_hub_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Path(tmp);source=c/'split.msh.h5';source.touch()
            config=dict(am.DEFAULTS,patches=dict(am.DEFAULTS['patches'],propeller='propeller_blade',propeller_hub='propeller_hub'))
            (c/'split.json').write_text(json.dumps(config))
            actual=am.read_config(c,source)
            self.assertEqual(actual['patches']['propeller'],'propeller_blade')
            self.assertEqual(actual['patches']['propeller_hub'],'propeller_hub')
            self.assertNotIn('propeller_hub',am.DEFAULTS['patches'])

    def test_all_nine_templates_keep_hub_boundary_conditions(self):
        import shutil
        root=Path(__file__).resolve().parents[1]
        templates=list((root/'CoreTemplates').glob('*/*/*/system/controlDict'))
        self.assertEqual(len(templates),9)
        for control in templates:
            with self.subTest(template=str(control)), tempfile.TemporaryDirectory() as tmp:
                c=Path(tmp)
                shutil.copytree(control.parents[1]/'0',c/'0')
                (c/'system').mkdir()
                shutil.copy2(control,c/'system/controlDict')
                am.configure_separate_hub(c)
                for field in (c/'0').iterdir():
                    if field.is_file() and 'boundaryField' in field.read_text(encoding='utf-8'):
                        self.assertIn('propeller_hub',field.read_text(encoding='utf-8'))
                self.assertIn('patch propeller;', (c/'system/controlDict').read_text())

    def test_hub_keeps_moving_wall_and_total_force_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Path(tmp);(c/'0').mkdir();(c/'system').mkdir()
            for name,kind in [('U','movingWallVelocity'),('p','zeroGradient'),('k','kqRWallFunction'),('omega','omegaWallFunction'),('nut','nutkWallFunction')]:
                (c/'0'/name).write_text('FoamFile {}\nboundaryField\n{\npropeller\n{type '+kind+'; value uniform 0;}\n}\n')
            (c/'system/controlDict').write_text('forcesBlades { patches (propeller); }\nwritePatchFields { patches (propeller); }\npropellerYPlusMean {patch propeller; fields (yPlus);}\n')
            am.configure_separate_hub(c)
            once=(c/'0/U').read_text()
            self.assertEqual(once.count('movingWallVelocity'),2)
            self.assertIn('propeller_hub',once)
            text=(c/'system/controlDict').read_text()
            self.assertEqual(text.count('patches (propeller propeller_hub)'),2)
            self.assertIn('patch propeller;',text)
            am.configure_separate_hub(c)
            self.assertEqual(once,(c/'0/U').read_text())


if __name__=='__main__':unittest.main()
