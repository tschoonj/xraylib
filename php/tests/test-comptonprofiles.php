<? 
include("xrltest.php");
include("xraylib.php");

class TestComptonProfiles extends XrlTest {
	function test_pz_0() {
		$profile = ComptonProfile(26, 0.0);
		assertAlmostEqual(7.060, $profile, 1E-6);
		$profile = ComptonProfile_Partial(26, N1_SHELL, 0.0);
		assertAlmostEqual(1.550, $profile, 1E-6);
		$profile1 = ComptonProfile_Partial(26, L2_SHELL, 0.0);
		$profile2 = ComptonProfile_Partial(26, L3_SHELL, 0.0);
		assertAlmostEqual($profile1, $profile2, 1E-6);
	}

	function test_pz_100() {
		$profile = ComptonProfile(26, 100.0);
		assertAlmostEqual(1.8E-05, $profile, 1E-8);
		$profile = ComptonProfile_Partial(26, N1_SHELL, 100.0);
		assertAlmostEqual(5.1E-09, $profile, 1E-12);
		$profile1 = ComptonProfile_Partial(26, L2_SHELL, 100.0);
		$profile2 = ComptonProfile_Partial(26, L3_SHELL, 100.0);
		assertAlmostEqual($profile1, $profile2, 1E-10);
		assertAlmostEqual($profile1, 1.100E-8, 1E-10);
	}

	function test_pz_50() {
		$profile = ComptonProfile(26, 50.0);
		assertAlmostEqual(0.0006843950273082384, $profile, 1E-8);
		$profile = ComptonProfile_Partial(26, N1_SHELL, 50.0);
		assertAlmostEqual(2.4322755767709126E-07, $profile, 1E-12);
		$profile1 = ComptonProfile_Partial(26, L2_SHELL, 50.0);
		$profile2 = ComptonProfile_Partial(26, L3_SHELL, 50.0);
		assertAlmostEqual($profile1, $profile2, 1E-10);
		assertAlmostEqual($profile1, 2.026953933016568e-06, 1E-10);
	}
	
	function test_bad_input() {
		assertException(ValueError, "ComptonProfile", 0, 0.0);
		$profile = ComptonProfile(102, 0.0);
		assertException(ValueError, "ComptonProfile", 103, 0.0);
		assertException(ValueError, "ComptonProfile", 26, -1.0);
		assertException(ValueError, "ComptonProfile_Partial", 0, K_SHELL, 0.0);
		$profile = ComptonProfile_Partial(102, K_SHELL, 0.0);
		assertException(ValueError, "ComptonProfile_Partial", 103, K_SHELL, 0.0);
		assertException(ValueError, "ComptonProfile_Partial", 26, K_SHELL, -1.0);
		assertException(ValueError, "ComptonProfile_Partial", 26, -1, 0.0);
		assertException(ValueError, "ComptonProfile_Partial", 26, N2_SHELL, 0.0);
	}
}

$suite = new XrlTestSuite();
$suite->append(new TestComptonProfiles());
$suite->run();
?>
