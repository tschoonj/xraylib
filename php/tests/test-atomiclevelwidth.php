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
		try {
			$width = AtomicLevelWidth(185, K_SHELL);
		} catch (Exception $e) {
			assertEqual($e->getCode(), ValueError);
			return;
		}
		throw new Exception();
	}
	function test_bad_shell() {
		try {
			$width = AtomicLevelWidth(26, -5);
		} catch (Exception $e) {
			assertEqual($e->getCode(), ValueError);
			return;
		}
		throw new Exception();
	}
	function test_invalid_shell() {
		try {
			$width = AtomicLevelWidth(26, N3_SHELL);
		} catch (Exception $e) {
			assertEqual($e->getCode(), ValueError);
			return;
		}
		throw new Exception();
	}
}

$suite = new XrlTestSuite();
$suite->append(new TestAtomicLevelWidth());
$suite->run();

?>
