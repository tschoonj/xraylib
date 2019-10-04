import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import org.junit.jupiter.api.Test;
import com.github.tschoonj.xraylib.Xraylib;

public class TestComptonProfiles {

	@Test
	public void test_pz_0() {
		double profile = Xraylib.ComptonProfile(26, 0.0);
		assertEquals(profile, 7.060, 1E-6);

		profile = Xraylib.ComptonProfile_Partial(26, Xraylib.N1_SHELL, 0.0);
		assertEquals(profile, 1.550, 1E-6);

		double profile1 = Xraylib.ComptonProfile_Partial(26, Xraylib.L2_SHELL, 0.0);
		double profile2 = Xraylib.ComptonProfile_Partial(26, Xraylib.L3_SHELL, 0.0);
		assertEquals(profile1, profile2, 1E-6);
		assertEquals(profile1, 0.065, 1E-6);
	}

	@Test
	public void test_pz_100() {
		double profile = Xraylib.ComptonProfile(26, 100.0);
		assertEquals(profile, 1.8E-5, 1E-8);

		profile = Xraylib.ComptonProfile_Partial(26, Xraylib.N1_SHELL, 100.0);
		assertEquals(profile, 5.1E-9, 1E-12);

		double profile1 = Xraylib.ComptonProfile_Partial(26, Xraylib.L2_SHELL, 100.0);
		double profile2 = Xraylib.ComptonProfile_Partial(26, Xraylib.L3_SHELL, 100.0);
		assertEquals(profile1, profile2, 1E-10);
		assertEquals(profile1, 1.1E-8, 1E-10);
	}

	@Test
	public void test_pz_50() {
		double profile = Xraylib.ComptonProfile(26, 50.0);
		assertEquals(profile, 0.0006843950273082384, 1E-8);

		profile = Xraylib.ComptonProfile_Partial(26, Xraylib.N1_SHELL, 50.0);
		assertEquals(profile, 2.4322755767709126e-07, 1E-12);

		double profile1 = Xraylib.ComptonProfile_Partial(26, Xraylib.L2_SHELL, 50.0);
		double profile2 = Xraylib.ComptonProfile_Partial(26, Xraylib.L3_SHELL, 50.0);
		assertEquals(profile1, profile2, 1E-10);
		assertEquals(profile1, 2.026953933016568e-06, 1E-10);
	}

	@Test
	public void test_bad_input() {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double profile = Xraylib.ComptonProfile(0, 0.0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		Xraylib.ComptonProfile(102, 0.0);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double profile = Xraylib.ComptonProfile(103, 0.0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double profile = Xraylib.ComptonProfile(26, -1.0);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_PZ);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double profile = Xraylib.ComptonProfile(26, 101);
		});
		assertEquals(exc.getMessage(), Xraylib.SPLINT_X_TOO_HIGH);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double profile = Xraylib.ComptonProfile_Partial(0, Xraylib.K_SHELL, 0.0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		Xraylib.ComptonProfile_Partial(102, Xraylib.K_SHELL, 0.0);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double profile = Xraylib.ComptonProfile_Partial(103, Xraylib.K_SHELL, 0.0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double profile = Xraylib.ComptonProfile_Partial(26, Xraylib.K_SHELL, -1.0);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_PZ);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double profile = Xraylib.ComptonProfile_Partial(26, -1, 0.0);
		});
		assertEquals(exc.getMessage(), Xraylib.INVALID_SHELL);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double profile = Xraylib.ComptonProfile_Partial(26, Xraylib.N2_SHELL, 0.0);
		});
		assertEquals(exc.getMessage(), Xraylib.INVALID_SHELL);
		exc = assertThrows(IllegalArgumentException.class, () -> {
			double profile = Xraylib.ComptonProfile_Partial(26, Xraylib.K_SHELL, 101);
		});
		assertEquals(exc.getMessage(), Xraylib.SPLINT_X_TOO_HIGH);

	}
}
