<? 
include("xrltest.php");
include("xraylib.php");

class TestAtomicLevelWidth extends XrlTest {
	function test_Fe_K() {
		$width = AtomicLevelWidth(26, K_SHELL);
		assertAlmostEqual($width, 1.19E-3);
	}
	function test_U_N7() {
		$width = AtomicLevelWidth(92, N7_SHELL);
		assertAlmostEqual($width, 0.31E-3);
	}
	function test_bad_Z() {
		assertException(ValueError, "AtomicLevelWidth", 185, K_SHELL);
	}
	function test_bad_shell() {
		assertException(ValueError, "AtomicLevelWidth", 26, -5);
	}
	function test_invalid_shell() {
		assertException(ValueError, "AtomicLevelWidth", 26, N3_SHELL);
	}
}

$suite = new XrlTestSuite();
$suite->append(new TestAtomicLevelWidth());
$suite->run();

?>
