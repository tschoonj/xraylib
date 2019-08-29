import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import org.junit.jupiter.api.Test;
import com.github.tschoonj.xraylib.Xraylib;

public class TestAtomicWeight {
	@Test
	public void test_Fe() {
		double weight = Xraylib.AtomicWeight(26);
		assertEquals(weight, 55.850, 1E-6);
	}

	@Test
	public void test_U() {
		double weight = Xraylib.AtomicWeight(92);
		assertEquals(weight, 238.070);
	}

	@Test
	public void test_bad_Z() {
		assertThrows(IllegalArgumentException.class, () -> {
			double width = Xraylib.AtomicWeight(185);
		}, Xraylib.Z_OUT_OF_RANGE);
	}
}
