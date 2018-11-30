<? 
include("xrltest.php");
include("xraylib.php");

class TestRadionuclides extends XrlTest {
	function test_good() {
		$list = GetRadioNuclideDataList();
		assertEqual(count($list), 10);
		foreach (array_values($list) as $i => $val) {
			$rnd = GetRadioNuclideDataByIndex($i);
			assertEqual($rnd['name'], $val);
			$rnd = GetRadioNuclideDataByName($val);
			assertEqual($rnd['name'], $val);
		}

		$rnd = GetRadioNuclideDataByIndex(3);
		assertEqual($rnd['A'], 125);
		assertAlmostEqual($rnd['GammaEnergies'], array(35.4919));
		assertAlmostEqual($rnd['GammaIntensities'], array(0.0668));
		assertEqual($rnd['N'], 72);
		assertAlmostEqual($rnd['XrayIntensities'], array(0.0023, 0.00112, 0.0063, 0.056, 0.035, 0.0042, 0.007, 0.00043, 0.0101, 0.0045, 0.00103, 0.0016, 3.24e-05, 0.406, 0.757, 0.0683, 0.132, 0.00121, 0.0381, 0.0058));
		assertEqual($rnd['XrayLines'], array(-86, -60, -89, -90, -63, -33, -34, -91, -95, -68, -38, -39, -1, -2, -3, -5, -6, -8, -11, -13));
		assertEqual($rnd['Z'], 53);
		assertEqual($rnd['Z_xray'], 52);
		assertEqual($rnd['nGammas'], 1);
		assertEqual($rnd['nXrays'], 20);
		assertEqual($rnd['name'], "125I");

		$rnd = GetRadioNuclideDataByName("125I");
		assertEqual($rnd['A'], 125);
		assertAlmostEqual($rnd['GammaEnergies'], array(35.4919));
		assertAlmostEqual($rnd['GammaIntensities'], array(0.0668));
		assertEqual($rnd['N'], 72);
		assertAlmostEqual($rnd['XrayIntensities'], array(0.0023, 0.00112, 0.0063, 0.056, 0.035, 0.0042, 0.007, 0.00043, 0.0101, 0.0045, 0.00103, 0.0016, 3.24e-05, 0.406, 0.757, 0.0683, 0.132, 0.00121, 0.0381, 0.0058));
		assertEqual($rnd['XrayLines'], array(-86, -60, -89, -90, -63, -33, -34, -91, -95, -68, -38, -39, -1, -2, -3, -5, -6, -8, -11, -13));
		assertEqual($rnd['Z'], 53);
		assertEqual($rnd['Z_xray'], 52);
		assertEqual($rnd['nGammas'], 1);
		assertEqual($rnd['nXrays'], 20);
		assertEqual($rnd['name'], "125I");
	}
	function test_bad() {
		assertException(ValueError, "GetRadioNuclideDataByName", "jjwqfejfjf");
		assertException(ValueError, "GetRadioNuclideDataByName", 0);
		assertException(ValueError, "GetRadioNuclideDataByName", NULL);
		assertException(ValueError, "GetRadioNuclideDataByIndex", -1);
		assertException(ValueError, "GetRadioNuclideDataByIndex", 10);
	}
}

$suite = new XrlTestSuite();
$suite->append(new TestRadionuclides());
$suite->run();

?>

