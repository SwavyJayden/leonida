"""Synthetic exit-code checks for the conformance expect command."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "conformance" / "compare.py"
SPEC = importlib.util.spec_from_file_location("conformance_expect_status", SCRIPT)
C = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = C
SPEC.loader.exec_module(C)


class ExpectStatusTest(unittest.TestCase):
    def test_input_errors_take_precedence_over_pixel_mismatches(self):
        for state, expected in (("matching", 0), ("different", 1), ("missing", 2), ("corrupt", 2)):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                image = root / "image.png"
                if state in ("matching", "different"):
                    C.write_png(str(image), C.make_solid_raster(1, 1, 2, (10, 20, 30), "synthetic"))
                elif state == "corrupt":
                    image.write_bytes(b"not a PNG")
                scenes = [{
                    "name": "scene", "png": image.name,
                    "checks": [{"x": 0, "y": 0, "r": 11 if state == "different" else 10,
                                "g": 20, "b": 30, "a": 255}],
                }]
                if expected == 2:
                    # A valid image with a mismatch must not hide an unreadable scene.
                    good = root / "good.png"
                    C.write_png(str(good), C.make_solid_raster(1, 1, 2, (10, 20, 30), "synthetic"))
                    scenes.append({"name": "different", "png": good.name,
                                   "checks": [{"x": 0, "y": 0, "r": 11, "g": 20, "b": 30, "a": 255}]})
                manifest = root / "manifest.json"
                C.write_selftest_manifest(str(manifest), "synthetic", "CPU", scenes)
                output = io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                    result = C.main(["expect", "--manifest", str(manifest)])
                self.assertEqual(result, expected, output.getvalue())
                if expected == 2:
                    self.assertIn("image problem", output.getvalue())
                    self.assertIn("different", output.getvalue())


if __name__ == "__main__":
    unittest.main()
