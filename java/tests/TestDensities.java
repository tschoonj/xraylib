import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import org.junit.jupiter.api.Test;
import com.github.tschoonj.xraylib.Xraylib;

public class TestDensities {

	@Test
	public void test_good_values() {

		assertEquals(Xraylib.ElementDensity(1), 0.000084, 1E-8);
		assertEquals(Xraylib.ElementDensity(50), 7.31, 1E-6);
		assertEquals(Xraylib.ElementDensity(98), 10.0, 1E-6);
	}

	@Test
	public void test_bad_values() {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double density = Xraylib.ElementDensity(0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		assertThrows(IllegalArgumentException.class, () -> {
			double density = Xraylib.ElementDensity(99);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);
	}
}
