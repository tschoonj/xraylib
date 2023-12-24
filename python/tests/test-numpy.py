import os
if "EXTRA_DLL_SEARCH_PATHS" in os.environ and hasattr(os, "add_dll_directory"):
    for path in os.environ["EXTRA_DLL_SEARCH_PATHS"].split(os.pathsep):
        os.add_dll_directory(path)
import unittest
import xraylib
import numpy as np



class TestNumpy(unittest.TestCase):
    def _test_np(self, dtype):
        for Z in np.arange(10, 20, dtype=dtype):
            print( xraylib.LineEnergy(Z, xraylib.KL2_LINE))

    def test_np_i16(self):
        self._test_np(np.int16)

    def test_np_i32(self):
        self._test_np(np.int32)

    def test_np_i64(self):
        self._test_np(np.int64)

if __name__ == '__main__':
    unittest.main(verbosity=2)
