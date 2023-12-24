import os
if "EXTRA_DLL_SEARCH_PATHS" in os.environ and hasattr(os, "add_dll_directory"):
    for path in os.environ["EXTRA_DLL_SEARCH_PATHS"].split(os.pathsep):
        os.add_dll_directory(path)
import unittest
import xraylib


class TestAtomicWeight(unittest.TestCase):
    def test_Fe(self):
        weight = xraylib.AtomicWeight(26)
        self.assertAlmostEqual(weight, 55.850)

    def test_U(self):
        weight = xraylib.AtomicWeight(92)
        self.assertAlmostEqual(weight, 238.070)

    def test_bad_Z(self):
        with self.assertRaises(ValueError):
            width = xraylib.AtomicWeight(185)

if __name__ == '__main__':
    unittest.main(verbosity=2)

