import os
if "EXTRA_DLL_SEARCH_PATHS" in os.environ and hasattr(os, "add_dll_directory"):
    for path in os.environ["EXTRA_DLL_SEARCH_PATHS"].split(os.pathsep):
        os.add_dll_directory(path)
import unittest
import xraylib


class TestNISTCompounds(unittest.TestCase):
    def test_good(self):
        list = xraylib.GetCompoundDataNISTList()
        self.assertEqual(len(list), 180)
        for i, v in enumerate(list):
            cdn = xraylib.GetCompoundDataNISTByIndex(i)
            self.assertEqual(cdn['name'], v)
            cdn = xraylib.GetCompoundDataNISTByName(v)
            self.assertEqual(cdn['name'], v)

        cdn = xraylib.GetCompoundDataNISTByIndex(5)
        self.assertEqual(cdn['nElements'], 4)
        self.assertEqual(cdn['density'], 0.001205)
        self.assertEqual(cdn['Elements'], (6, 7, 8, 18))
        self.assertAlmostEqual(cdn['massFractions'], (0.000124, 0.755267, 0.231781, 0.012827))
        self.assertEqual(cdn['name'], 'Air, Dry (near sea level)')

        cdn = xraylib.GetCompoundDataNISTByName('Air, Dry (near sea level)')
        self.assertEqual(cdn['nElements'], 4)
        self.assertEqual(cdn['density'], 0.001205)
        self.assertEqual(cdn['Elements'], (6, 7, 8, 18))
        self.assertAlmostEqual(cdn['massFractions'], (0.000124, 0.755267, 0.231781, 0.012827))
        self.assertEqual(cdn['name'], 'Air, Dry (near sea level)')

    def test_bad(self):
        with self.assertRaises(ValueError):
            xraylib.GetCompoundDataNISTByName("jwjfpfj")
        with self.assertRaises(TypeError):
            xraylib.GetCompoundDataNISTByName(0)
        with self.assertRaises(ValueError):
            xraylib.GetCompoundDataNISTByName(None)
        with self.assertRaises(ValueError):
            xraylib.GetCompoundDataNISTByIndex(-1)
        with self.assertRaises(ValueError):
            xraylib.GetCompoundDataNISTByIndex(180)
        with self.assertRaises(TypeError):
            xraylib.GetCompoundDataNISTByIndex(None)
        with self.assertRaises(TypeError):
            xraylib.GetCompoundDataNISTByIndex("jpwjfpfwj")

if __name__ == '__main__':
    unittest.main(verbosity=2)
