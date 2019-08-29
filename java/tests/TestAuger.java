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
		assertThrows(IllegalArgumentException.class, () -> {
			double rate = Xraylib.AugerRate(-35, Xraylib.L3_M4N7_AUGER);
		}, Xraylib.Z_OUT_OF_RANGE);
	}

	@Test
	public void test_rate_bad_trans() {
		assertThrows(IllegalArgumentException.class, () -> {
			double rate = Xraylib.AugerRate(82, Xraylib.M4_M5Q3_AUGER + 1);
		}, Xraylib.UNKNOWN_AUGER);
	}

	@Test
	public void test_rate_invalid_trans() {
		assertThrows(IllegalArgumentException.class, () -> {
			double rate = Xraylib.AugerRate(62, Xraylib.L3_M4N7_AUGER);
		}, Xraylib.INVALID_AUGER);
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
	public void test_yield_bad_Z() {
		assertThrows(IllegalArgumentException.class, () -> {
			double ayield = Xraylib.AugerYield(-35, Xraylib.K_SHELL);
		}, Xraylib.Z_OUT_OF_RANGE);
	}

	@Test
	public void test_yield_invalid_shell() {
		assertThrows(IllegalArgumentException.class, () -> {
			double ayield = Xraylib.AugerYield(82, Xraylib.N2_SHELL);
		}, Xraylib.INVALID_SHELL);
	}

	@Test
	public void test_yield_bad_shell() {
		assertThrows(IllegalArgumentException.class, () -> {
			double ayield = Xraylib.AugerYield(82, -5);
		}, Xraylib.UNKNOWN_SHELL);
	}
}
