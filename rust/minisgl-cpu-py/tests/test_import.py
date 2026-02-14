import importlib
import unittest


class RustBindingImportTests(unittest.TestCase):
    def test_import_extension_module(self) -> None:
        mod = importlib.import_module("minisgl_cpu")
        self.assertEqual(mod.ping(), "ok")
        self.assertTrue(hasattr(mod, "SamplingParams"))


if __name__ == "__main__":
    unittest.main()
