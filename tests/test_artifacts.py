import tempfile
import unittest
from pathlib import Path

from reviewwise.artifacts import check_file, load_manifest, versions, write_json
from reviewwise.config import FEATURES, MODEL_NAMES, MODEL_PARAMETERS, POLICY
from reviewwise.data import fingerprint


class ArtifactTests(unittest.TestCase):
    def test_data_features_policy_and_model_hash_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            data = folder / "input.csv"
            data.write_text("original", encoding="utf-8")
            hashes = {}
            for name in MODEL_NAMES:
                file = folder / f"{name}.joblib"
                file.write_bytes(b"local test placeholder - never deserialized")
                hashes[name] = fingerprint(file)
            manifest = {
                "schema_version": 1, "features": FEATURES, "model_parameters": MODEL_PARAMETERS,
                "policy": POLICY, "selected_model": "logistic_regression",
                "input_sha256": fingerprint(data), "versions": versions(), "model_files": hashes,
            }
            write_json(folder / "manifest.json", manifest)
            self.assertEqual(load_manifest(folder, data)["selected_model"], "logistic_regression")
            data.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_manifest(folder, data)
            data.write_text("original", encoding="utf-8")
            write_json(folder / "manifest.json", {**manifest, "features": FEATURES + ["Class"]})
            with self.assertRaisesRegex(ValueError, "feature"):
                load_manifest(folder, data)
            write_json(folder / "manifest.json", {**manifest, "policy": {}})
            with self.assertRaisesRegex(ValueError, "Configuration"):
                load_manifest(folder, data)
            with self.assertRaisesRegex(ValueError, "changed artifact"):
                check_file(data, "incorrect digest")


if __name__ == "__main__":
    unittest.main()
