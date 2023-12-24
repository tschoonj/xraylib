import os
if "EXTRA_DLL_SEARCH_PATHS" in os.environ and hasattr(os, "add_dll_directory"):
    for path in os.environ["EXTRA_DLL_SEARCH_PATHS"].split(os.pathsep):
        os.add_dll_directory(path)
import unittest
import xraylib


class TestAtomicLevelWidth(unittest.TestCase):
    def test_Fe_K(self):
        width = xraylib.AtomicLevelWidth(26, xraylib.K_SHELL)
        self.assertAlmostEqual(width, 1.19E-3)

    def test_U_N7(self):
        width = xraylib.AtomicLevelWidth(92, xraylib.N7_SHELL)
        self.assertAlmostEqual(width, 0.31E-3)

    def test_bad_Z(self):
        with self.assertRaises(ValueError):
            width = xraylib.AtomicLevelWidth(185, xraylib.K_SHELL)

    def test_bad_shell(self):
        with self.assertRaises(ValueError):
            width = xraylib.AtomicLevelWidth(26, -5)

    def test_invalid_shell(self):
        with self.assertRaises(ValueError):
            width = xraylib.AtomicLevelWidth(26, xraylib.N3_SHELL)

if __name__ == '__main__':
    unittest.main(verbosity=2)
