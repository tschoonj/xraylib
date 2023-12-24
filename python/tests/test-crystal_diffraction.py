import os
if "EXTRA_DLL_SEARCH_PATHS" in os.environ and hasattr(os, "add_dll_directory"):
    for path in os.environ["EXTRA_DLL_SEARCH_PATHS"].split(os.pathsep):
        os.add_dll_directory(path)
import unittest
import xraylib
import math


class TestCrystalDiffraction(unittest.TestCase):
    def test_crystal_diffraction(self):
        crystals_list = xraylib.Crystal_GetCrystalsList()
        self.assertEqual(len(crystals_list), 38)
        for crystal_name in crystals_list:
            cs = xraylib.Crystal_GetCrystal(crystal_name)
            self.assertEqual(crystal_name, cs['name'])

        with self.assertRaises(ValueError):
            cs = xraylib.Crystal_GetCrystal(None)

        with self.assertRaises(ValueError):
            cs = xraylib.Crystal_GetCrystal("non-existent-crystal")

        cs = xraylib.Crystal_GetCrystal("Diamond") 

        cs_copy = xraylib.Crystal_MakeCopy(cs)

        with self.assertRaises(ValueError):
            xraylib.Crystal_AddCrystal(cs)

        with self.assertRaises(ValueError):
            xraylib.Crystal_AddCrystal(cs_copy)

        cs_copy['name'] = "Diamond-copy"
        xraylib.Crystal_AddCrystal(cs_copy)

        cs_copy['name'] = 20012016
        with self.assertRaises(TypeError):
            xraylib.Crystal_AddCrystal(cs_copy)

        cs_copy['name'] = "Diamond-copy"

        cs_copy['atom'] = list()
        with self.assertRaises(TypeError):
            xraylib.Crystal_AddCrystal(cs_copy)

        cs_copy['atom'] = (25, "jkewjfpwejffj", None, )
        with self.assertRaises(TypeError):
            xraylib.Crystal_AddCrystal(cs_copy)

        del cs_copy['atom']

        with self.assertRaises(KeyError):
            xraylib.Crystal_AddCrystal(cs_copy)

        crystals_list = xraylib.Crystal_GetCrystalsList()
        self.assertEqual(len(crystals_list), 39)

        for crystal_name in crystals_list:
            cs = xraylib.Crystal_GetCrystal(crystal_name)
            self.assertEqual(crystal_name, cs['name'])

        current_ncrystals = len(crystals_list)

        for i in range(xraylib.CRYSTALARRAY_MAX):
            cs_copy = xraylib.Crystal_MakeCopy(cs)
            cs_copy['name'] = "Diamond copy {}".format(i)
            if current_ncrystals < xraylib.CRYSTALARRAY_MAX:
                xraylib.Crystal_AddCrystal(cs_copy)
                current_ncrystals = current_ncrystals + 1
                self.assertEqual(len(xraylib.Crystal_GetCrystalsList()), current_ncrystals)
            else:
                with self.assertRaises(RuntimeError):
                    xraylib.Crystal_AddCrystal(cs_copy)
                self.assertEqual(len(xraylib.Crystal_GetCrystalsList()), xraylib.CRYSTALARRAY_MAX)

        cs = xraylib.Crystal_GetCrystal("Diamond") 

        # Bragg angle
        angle = xraylib.Bragg_angle(cs, 10.0, 1, 1, 1)
        self.assertAlmostEqual(angle, 0.3057795845795849)

        with self.assertRaises(TypeError):
            angle = xraylib.Bragg_angle(None, 10.0, 1, 1, 1)

        with self.assertRaises(ValueError):
            angle = xraylib.Bragg_angle(cs, -10.0, 1, 1, 1)

        with self.assertRaises(TypeError):
            angle = xraylib.Bragg_angle(cs, 1, 1, 1)

        
	# Q_scattering_amplitude
        tmp = xraylib.Q_scattering_amplitude(cs, 10.0, 1, 1, 1, math.pi/4.0)
        self.assertAlmostEqual(tmp, 0.19184445408324474)

        tmp = xraylib.Q_scattering_amplitude(cs, 10.0, 0, 0, 0, math.pi/4.0)
        self.assertEqual(tmp, 0.0)

        # Atomic factors
        (f0, f_prime, f_prime2) = xraylib.Atomic_Factors(26, 10.0, 1.0, 10.0)
        self.assertAlmostEqual(f0, 65.15)
        self.assertAlmostEqual(f_prime, -0.22193271025027966)
        self.assertAlmostEqual(f_prime2, 22.420270655080493)

        with self.assertRaises(ValueError):
            (f0, f_prime, f_prime2) = xraylib.Atomic_Factors(-10, 10.0, 1.0, 10.0)

        # unit cell volume
        tmp = xraylib.Crystal_UnitCellVolume(cs)
        self.assertAlmostEqual(tmp, 45.376673902751)

        # crystal dspacing
        tmp = xraylib.Crystal_dSpacing(cs, 1, 1, 1)
        self.assertAlmostEqual(tmp, 2.0592870875248344)

        del cs

if __name__ == '__main__':
    unittest.main(verbosity=2)
