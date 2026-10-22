import copy
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from validate_arkit_recording import validate, export_colmap


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'images').mkdir()
        Image.new('RGB', (16, 12)).save(self.root / 'images/00000.jpg')
        self.data = dict(schema_version=1, source='arkit_rgb', matrix_layout='column_major',
                         pose='camera_to_world', world_units='meters', frames=[dict(
                             image='images/00000.jpg', width=16, height=12, timestamp_seconds=1,
                             tracking='normal', intrinsics=[10,0,0,0,10,0,8,6,1],
                             camera_to_world=[1,0,0,0,0,1,0,0,0,0,1,0,1,2,3,1])])

    def write(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.data))

    def test_valid(self):
        self.write()
        self.assertEqual(len(validate(self.root)['frames']), 1)

    def test_path_escape(self):
        self.data['frames'][0]['image'] = '../outside.jpg'
        self.write()
        with self.assertRaises(ValueError): validate(self.root)

    def test_duplicate_timestamp(self):
        second = copy.deepcopy(self.data['frames'][0])
        second['image'] = 'images/00001.jpg'
        self.data['frames'].append(second)
        self.write()
        with self.assertRaises(ValueError): validate(self.root)

    def test_scaled_pose_rejected(self):
        self.data['frames'][0]['camera_to_world'][0] = 2
        self.write()
        with self.assertRaises(ValueError): validate(self.root)

    def test_resolution_rejected(self):
        self.data['frames'][0]['width'] = 12
        self.write()
        with self.assertRaises(ValueError): validate(self.root)

    def test_colmap_pose_axes(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('COLMAP conversion requires numpy')
        export_colmap(self.data, self.root / 'sparse')
        fields = (self.root / 'sparse/images.txt').read_text().splitlines()[0].split()
        w, x, y, z = map(float, fields[1:5])
        rotation = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        np.testing.assert_allclose(rotation, np.diag([1,-1,-1]), atol=1e-10)
        np.testing.assert_allclose(list(map(float, fields[5:8])), [-1,2,3])


if __name__ == '__main__': unittest.main()
