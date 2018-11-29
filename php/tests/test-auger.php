<? 
include("xrltest.php");
include("xraylib.php");

class TestAugerRate extends XrlTest {
	function test_Pb_K_L3M5() {
		$rate = AugerRate(82, K_L3M5_AUGER);
		assertAlmostEqual($rate, 0.004573193387);
	}
	function test_Pb_L3_M4N7() {
		$rate = AugerRate(82, L3_M4N7_AUGER);
		assertAlmostEqual($rate, 0.0024327572005);
	}
	function test_bad_Z() {
		assertException(ValueError, "AugerRate", -35, L3_M4N7_AUGER);
	}
	function test_bad_trans() {
		assertException(ValueError, "AugerRate", 82, M4_M5Q3_AUGER);
	}
	function test_invalid_trans() {
		assertException(ValueError, "AugerRate", 62, L3_M4N7_AUGER);
	}
}

class TestAugerYield extends XrlTest {
	function test_Pb_K() {
		$ayield = AugerYield(82, K_SHELL);
		assertAlmostEqual($ayield, 1.0 - FluorYield(82, K_SHELL));
	}
	function test_Pb_M3() {
		$ayield = AugerYield(82, M3_SHELL);
		assertAlmostEqual($ayield, 0.1719525);
	}
	function test_bad_Z() {
		assertException(ValueError, "AugerYield", -35, K_SHELL);
	}
	function test_bad_shell() {
		assertException(ValueError, "AugerYield", 82, N2_SHELL);
	}
}

$suite = new XrlTestSuite();
$suite->append(new TestAugerRate());
$suite->append(new TestAugerYield());
$suite->run();

?>

