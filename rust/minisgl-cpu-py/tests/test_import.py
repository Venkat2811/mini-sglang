import importlib
import unittest
from array import array


class RustBindingImportTests(unittest.TestCase):
    def test_import_extension_module(self) -> None:
        mod = importlib.import_module("minisgl_cpu")
        self.assertEqual(mod.ping(), "ok")
        self.assertTrue(hasattr(mod, "SamplingParams"))
        self.assertTrue(hasattr(mod, "RadixCacheManager"))
        positions = mod.make_positions([1, 2], [3, 5])
        self.assertEqual(positions, [1, 2, 2, 3, 4])
        mapping = mod.make_input_mapping([7, 9], [1, 2], [3, 5])
        self.assertEqual(mapping, [7, 7, 9, 9, 9])
        packed = mod.make_metadata_buffers(
            [7, 9], [1, 2], [3, 5], [7, 9], [3, 5], [True, False]
        )
        self.assertEqual(len(packed), 4)
        positions_buf = array("i")
        positions_buf.frombytes(packed[0])
        self.assertEqual(list(positions_buf), [1, 2, 2, 3, 4])
        admitted = mod.prefill_admission_plan(
            token_budget=4,
            reserved_size=0,
            cache_available_size=32,
            table_available_size=1,
            input_len=6,
            output_len=4,
            cached_len=2,
        )
        self.assertTrue(admitted[0])


if __name__ == "__main__":
    unittest.main()
