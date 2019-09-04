import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import org.junit.jupiter.api.Test;
import com.github.tschoonj.xraylib.Xraylib;

public class TestAuger {
	@Test
	public void test_rate_Pb_K_L3M5() {
		double rate = Xraylib.AugerRate(82, Xraylib.K_L3M5_AUGER);
		assertEquals(rate, 0.004573193387, 1E-6);
	}

	@Test
	public void test_rate_Pb_L3_M4N7() {
		double rate = Xraylib.AugerRate(82, Xraylib.L3_M4N7_AUGER);
		assertEquals(rate, 0.0024327572005, 1E-6);
	}

	@Test
	public void test_rate_bad_Z() {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double rate = Xraylib.AugerRate(-35, Xraylib.L3_M4N7_AUGER);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);
	}

	@Test
	public void test_rate_bad_trans() {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double rate = Xraylib.AugerRate(82, Xraylib.M4_M5Q3_AUGER + 1);
		});
		assertEquals(exc.getMessage(), Xraylib.UNKNOWN_AUGER);
	}

	@Test
	public void test_rate_invalid_trans() {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double rate = Xraylib.AugerRate(62, Xraylib.L3_M4N7_AUGER);
		});
		assertEquals(exc.getMessage(), Xraylib.INVALID_AUGER);
	}

	@Test
	public void test_yield_Pb_K() {
		double ayield = Xraylib.AugerYield(82, Xraylib.K_SHELL);
		assertEquals(ayield, 1.0 - Xraylib.FluorYield(82, Xraylib.K_SHELL));
	}

	@Test
	public void test_yield_Pb_M3() {
		double ayield = Xraylib.AugerYield(82, Xraylib.M3_SHELL);
		assertEquals(ayield, 0.1719525, 1E-6);
	}

	@Test
	public void test_yield_Pb_L1() {
		double ayield = Xraylib.AugerYield(82, Xraylib.L1_SHELL);
		assertEquals(ayield, 0.1825, 1E-6);
	}

	@Test
	public void test_yield_bad_Z() {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double ayield = Xraylib.AugerYield(-35, Xraylib.K_SHELL);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);
	}

	@Test
	public void test_yield_invalid_shell() {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double ayield = Xraylib.AugerYield(26, Xraylib.M5_SHELL);
		});
		assertEquals(exc.getMessage(), Xraylib.INVALID_SHELL);
	}

	@Test
	public void test_yield_bad_shell() {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double ayield = Xraylib.AugerYield(82, -5);
		});
		assertEquals(exc.getMessage(), Xraylib.UNKNOWN_SHELL);
	}
}
