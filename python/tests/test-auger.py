import os
if "EXTRA_DLL_SEARCH_PATHS" in os.environ and hasattr(os, "add_dll_directory"):
    for path in os.environ["EXTRA_DLL_SEARCH_PATHS"].split(os.pathsep):
        os.add_dll_directory(path)
import unittest
import xraylib


class TestAugerRate(unittest.TestCase):
    def test_Pb_K_L3M5(self):
        rate = xraylib.AugerRate(82, xraylib.K_L3M5_AUGER)
        self.assertAlmostEqual(rate, 0.004573193387)

    def test_Pb_L3_M4N7(self):
        rate = xraylib.AugerRate(82, xraylib.L3_M4N7_AUGER)
        self.assertAlmostEqual(rate, 0.0024327572005)

    def test_bad_Z(self):
        with self.assertRaises(ValueError):
            rate = xraylib.AugerRate(-35, xraylib.L3_M4N7_AUGER)

    def test_bad_trans(self):
        with self.assertRaises(ValueError):
            rate = xraylib.AugerRate(82, xraylib.M4_M5Q3_AUGER + 1)

    def test_invalid_trans(self):
        with self.assertRaises(ValueError):
            rate = xraylib.AugerRate(62, xraylib.L3_M4N7_AUGER)

class TestAugerYield(unittest.TestCase):
    def test_Pb_K(self):
        ayield = xraylib.AugerYield(82, xraylib.K_SHELL)
        self.assertAlmostEqual(ayield, 1.0 - xraylib.FluorYield(82, xraylib.K_SHELL))

    def test_Pb_M3(self):
        ayield = xraylib.AugerYield(82, xraylib.M3_SHELL)
        self.assertAlmostEqual(ayield, 0.1719525)

    def test_bad_Z(self):
        with self.assertRaises(ValueError):
            ayield = xraylib.AugerYield(-35, xraylib.K_SHELL)

    def test_bad_shell(self):
        with self.assertRaises(ValueError):
            ayield = xraylib.AugerYield(82, xraylib.N2_SHELL)

if __name__ == '__main__':
    unittest.main(verbosity=2)
