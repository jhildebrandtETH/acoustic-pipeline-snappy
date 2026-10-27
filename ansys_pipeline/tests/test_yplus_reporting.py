import gzip
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.yplus_reporting import create_yplus_report, histogram_masks, class_masks, append_yplus_report
from tools.reporting import create_yplus_distribution_plot


class YPlusReportingTests(unittest.TestCase):
    def fixture(self, root, values='nonuniform List<scalar> 2 (10 40)'):
        root=Path(root)
        mesh=root/'constant/polyMesh'; mesh.mkdir(parents=True)
        (mesh/'boundary').write_text('FoamFile {format ascii;}\n1\n(\npropeller {type wall; nFaces 2; startFace 0;}\n)')
        (mesh/'points').write_text('FoamFile {format ascii;}\n4\n((0 0 0)(2 0 0)(0 1 0)(0 2 0))')
        (mesh/'faces').write_text('FoamFile {format ascii;}\n2\n(3\n(0\n1\n2) 3(0 1 3))')
        (root/'1').mkdir()
        (root/'1/yPlus').write_text('FoamFile {format ascii;}\nboundaryField { not_propeller {value uniform 999;} "propeller" {value '+values+';} }')
        return root

    def test_primary_area_and_diagnostic_faces_via_public_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=self.fixture(tmp)
            image,s=create_yplus_distribution_plot(c,c/'report','propeller')
            self.assertTrue(image.is_file())
            self.assertEqual(s['status'],'complete')
            self.assertEqual(s['average_yplus'],30)
            self.assertEqual(s['face_weighted']['mean'],25)
            self.assertEqual(s['area_weighted']['median'],40)
            np.testing.assert_allclose(s['area_weighted']['class_percent'],[0,100/3,200/3])
            self.assertEqual(s['face_weighted']['class_percent'],[0,50,50])
            self.assertEqual(json.loads((c/'report/yplus_distribution.json').read_text())['primary_weighting'],'surface_area')

    def test_uniform_gzip_and_snapshot_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=self.fixture(tmp,'uniform 20')
            (c/'1/polyMesh').mkdir()
            # Updated last vertex changes the second face area from 2 to 4.
            (c/'1/polyMesh/points.gz').write_bytes(gzip.compress(b'FoamFile {format ascii;}\n4\n((0 0 0)(2 0 0)(0 1 0)(0 4 0))'))
            f=c/'1/yPlus';f.with_suffix('.gz').write_bytes(gzip.compress(f.read_bytes()));f.unlink()
            _,s=create_yplus_report(c,c/'report','propeller')
            self.assertEqual(s['average_yplus'],20)
            self.assertEqual(s['patch_area_m2'],5)
            self.assertEqual(s['area_weighted']['class_percent'],[0,100,0])
            self.assertIn('1',s['mesh_sources']['points'])

    def test_thresholds_match_chart_and_summary(self):
        v=np.array([0,5,30,30.001,50,100,200,201])
        classes=class_masks(v); bins=histogram_masks(v)
        self.assertTrue(np.array_equal(classes[1],bins[1]))
        np.testing.assert_array_equal(np.sum(bins,axis=0),np.ones(len(v)))
        self.assertEqual(int(classes[1].sum()),2)

    def test_missing_area_never_becomes_face_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=self.fixture(tmp);(c/'constant/polyMesh/faces').unlink()
            _,s=create_yplus_report(c,c/'report','propeller')
            self.assertEqual(s['status'],'face_only')
            self.assertIsNone(s['average_yplus'])
            self.assertEqual(s['face_weighted']['mean'],25)
            self.assertIn('area_unavailable_reason',s)

    def test_invalid_count_and_nonfinite_rejected(self):
        for values in ['nonuniform List<scalar> 3 (10 40)', 'nonuniform List<scalar> 2 (10 nan)', 'uniform -1']:
            with self.subTest(values=values),tempfile.TemporaryDirectory() as tmp:
                c=self.fixture(tmp,values)
                image,s=create_yplus_report(c,c/'report','propeller')
                self.assertIsNone(image)
                self.assertEqual(s['status'],'unavailable')

    def test_newer_output_does_not_select_stale_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=self.fixture(tmp);(c/'2').mkdir()
            _,s=create_yplus_report(c,c/'report','propeller')
            self.assertEqual(s['status'],'unavailable')
            self.assertEqual(s['time_dir'],'2')
        with tempfile.TemporaryDirectory() as tmp:
            c=self.fixture(tmp);(c/'processor0/2').mkdir(parents=True)
            _,s=create_yplus_report(c,c/'report','propeller')
            self.assertEqual(s['status'],'unavailable')
            self.assertIn('Processor',s['reason'])

    def test_pdf_page_labels_weighting_and_scope(self):
        from reportlab.pdfgen.canvas import Canvas
        with tempfile.TemporaryDirectory() as tmp:
            c=self.fixture(tmp);image,s=create_yplus_report(c,c/'report','propeller')
            pdf=c/'page.pdf';canvas=Canvas(str(pdf),pageCompression=0);append_yplus_report(canvas,image,s);canvas.save()
            text=pdf.read_bytes().decode('latin1')
            self.assertIn('Area-weighted mean y+: 30.00',text)
            self.assertIn('Face-weighted mean y+: 25.00',text)
            self.assertIn('One latest reconstructed snapshot',text)
            self.assertIn('Surface area:',text)


if __name__=='__main__':
    unittest.main()
