<? 
include("xrltest.php");
include("xraylib.php");

class TestAtomicWeight extends XrlTest {
	function test_Fe() {
		$weight = AtomicWeight(26);
		assertAlmostEqual($weight, 55.850);
	}
	function test_U() {
		$weight = AtomicWeight(92);
		assertAlmostEqual($weight, 238.070);
	}
	function test_bad_Z() {
		try {
			$weight = AtomicWeight(185);
		} catch (Exception $e) {
			assertEqual($e->getCode(), ValueError);
			return;
		}
		throw new Exception();
	}
}

$suite = new XrlTestSuite();
$suite->append(new TestAtomicWeight());
$suite->run();

?>

