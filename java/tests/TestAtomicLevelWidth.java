import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import org.junit.jupiter.api.Test;
import com.github.tschoonj.xraylib.Xraylib;

public class TestAtomicLevelWidth {
	@Test
	public void test_Fe_K() {
		double width = Xraylib.AtomicLevelWidth(26, Xraylib.K_SHELL);
		assertEquals(width, 1.19E-3, 1E-6);
	}

	@Test
	public void test_U_N7() {
		double width = Xraylib.AtomicLevelWidth(92, Xraylib.N7_SHELL);
		assertEquals(width, 0.31E-3, 1E-6);
	}

	@Test
	public void test_bad_Z() {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double width = Xraylib.AtomicLevelWidth(185, Xraylib.K_SHELL);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);
	}

	@Test
	public void test_bad_shell() {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double width = Xraylib.AtomicLevelWidth(26, -5);
		});
		assertEquals(exc.getMessage(), Xraylib.UNKNOWN_SHELL);
	}

	@Test
	public void test_invalid_shell() {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double width = Xraylib.AtomicLevelWidth(26, Xraylib.N3_SHELL);
		});
		assertEquals(exc.getMessage(), Xraylib.INVALID_SHELL);
	}
}
