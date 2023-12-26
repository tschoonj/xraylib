import os
if "EXTRA_DLL_SEARCH_PATHS" in os.environ and hasattr(os, "add_dll_directory"):
    for path in os.environ["EXTRA_DLL_SEARCH_PATHS"].split(os.pathsep):
        os.add_dll_directory(path)
import unittest
import xraylib


class TestRadionuclides(unittest.TestCase):
    def test_good(self):
        list = xraylib.GetRadioNuclideDataList()
        self.assertEqual(len(list), 10)
        for i, v in enumerate(list):
            rnd = xraylib.GetRadioNuclideDataByIndex(i)
            self.assertEqual(rnd['name'], v)
            rnd = xraylib.GetRadioNuclideDataByName(v)
            self.assertEqual(rnd['name'], v)

        rnd = xraylib.GetRadioNuclideDataByIndex(3)
        self.assertEqual(rnd['A'], 125)
        self.assertAlmostEqual(rnd['GammaEnergies'], (35.4919, ))
        self.assertAlmostEqual(rnd['GammaIntensities'], (0.0668, ))
        self.assertEqual(rnd['N'], 72)
        XrayIntensities = (0.0023, 0.00112, 0.0063, 0.056, 0.035, 0.0042, 0.007, 0.00043, 0.0101, 0.0045, 0.00103, 0.0016, 3.24e-05, 0.406, 0.757, 0.0683, 0.132, 0.00121, 0.0381, 0.0058)
        self.assertAlmostEqual(rnd['XrayIntensities'], XrayIntensities)
        XrayLines = (-86, -60, -89, -90, -63, -33, -34, -91, -95, -68, -38, -39, -1, -2, -3, -5, -6, -8, -11, -13)
        self.assertEqual(rnd['XrayLines'], XrayLines)
        self.assertEqual(rnd['Z'], 53)
        self.assertEqual(rnd['Z_xray'], 52)
        self.assertEqual(rnd['nGammas'], 1)
        self.assertEqual(rnd['nXrays'], 20)
        self.assertEqual(rnd['name'], '125I')

        rnd = xraylib.GetRadioNuclideDataByName('125I')
        self.assertEqual(rnd['A'], 125)
        self.assertAlmostEqual(rnd['GammaEnergies'], (35.4919, ))
        self.assertAlmostEqual(rnd['GammaIntensities'], (0.0668, ))
        self.assertEqual(rnd['N'], 72)
        XrayIntensities = (0.0023, 0.00112, 0.0063, 0.056, 0.035, 0.0042, 0.007, 0.00043, 0.0101, 0.0045, 0.00103, 0.0016, 3.24e-05, 0.406, 0.757, 0.0683, 0.132, 0.00121, 0.0381, 0.0058)
        self.assertAlmostEqual(rnd['XrayIntensities'], XrayIntensities)
        XrayLines = (-86, -60, -89, -90, -63, -33, -34, -91, -95, -68, -38, -39, -1, -2, -3, -5, -6, -8, -11, -13)
        self.assertEqual(rnd['XrayLines'], XrayLines)
        self.assertEqual(rnd['Z'], 53)
        self.assertEqual(rnd['Z_xray'], 52)
        self.assertEqual(rnd['nGammas'], 1)
        self.assertEqual(rnd['nXrays'], 20)
        self.assertEqual(rnd['name'], '125I')

    def test_bad(self):
        with self.assertRaises(ValueError):
            xraylib.GetRadioNuclideDataByName("jwjfpfj")
        with self.assertRaises(TypeError):
            xraylib.GetRadioNuclideDataByName(0)
        with self.assertRaises(ValueError):
            xraylib.GetRadioNuclideDataByName(None)
        with self.assertRaises(ValueError):
            xraylib.GetRadioNuclideDataByIndex(-1)
        with self.assertRaises(ValueError):
            xraylib.GetRadioNuclideDataByIndex(10)
        with self.assertRaises(TypeError):
            xraylib.GetRadioNuclideDataByIndex(None)
        with self.assertRaises(TypeError):
            xraylib.GetRadioNuclideDataByIndex("jpwjfpfwj")

if __name__ == '__main__':
    unittest.main(verbosity=2)

