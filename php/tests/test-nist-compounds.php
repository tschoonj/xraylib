<? 
include("xrltest.php");
include("xraylib.php");

class TestNISTCompounds extends XrlTest {
	function test_good() {
		$list = GetCompoundDataNISTList();
		assertEqual(count($list), 180);
		foreach (array_values($list) as $i => $val) {
			$cdn = GetCompoundDataNISTByIndex($i);
			assertEqual($cdn['name'], $val);
			$cdn = GetCompoundDataNISTByName($val);
			assertEqual($cdn['name'], $val);
		}
		$cdn = GetCompoundDataNISTByIndex(5);
		assertEqual($cdn['nElements'], 4);
		assertAlmostEqual($cdn['density'], 0.001205);
		assertEqual($cdn['Elements'], array(6, 7, 8, 18));
		assertAlmostEqual($cdn['massFractions'], array(0.000124, 0.755267, 0.231781, 0.012827));
		assertEqual($cdn['name'], "Air, Dry (near sea level)");

		$cdn = GetCompoundDataNISTByName("Air, Dry (near sea level)");
		assertEqual($cdn['nElements'], 4);
		assertAlmostEqual($cdn['density'], 0.001205);
		assertEqual($cdn['Elements'], array(6, 7, 8, 18));
		assertAlmostEqual($cdn['massFractions'], array(0.000124, 0.755267, 0.231781, 0.012827));
		assertEqual($cdn['name'], "Air, Dry (near sea level)");
	}
	function test_bad() {
		assertException(ValueError, "GetCompoundDataNISTByName", "jjwqfejfjf");
		assertException(ValueError, "GetCompoundDataNISTByName", 0);
		assertException(ValueError, "GetCompoundDataNISTByName", NULL);
		assertException(ValueError, "GetCompoundDataNISTByIndex", -1);
		assertException(ValueError, "GetCompoundDataNISTByIndex", 180);
	}
}

$suite = new XrlTestSuite();
$suite->append(new TestNISTCompounds());
$suite->run();

?>
