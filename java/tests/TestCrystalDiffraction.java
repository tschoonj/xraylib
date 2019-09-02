import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
 
import com.github.tschoonj.xraylib.Xraylib;
import com.github.tschoonj.xraylib.Crystal_Struct;

public class TestCrystalDiffraction {

	private Crystal_Struct cs;

	@BeforeEach
	public void init() {
		cs = Xraylib.Crystal_GetCrystal("Diamond");
	}

	@Test
	public void test_list_crystals() {
        	String[] crystals_list = Xraylib.Crystal_GetCrystalsList();
		assertEquals(crystals_list.length, 38);
		for (String crystal_name : crystals_list) {
			Crystal_Struct cs = Xraylib.Crystal_GetCrystal(crystal_name);
			assertEquals(crystal_name, cs.name);
		}
	}

	@ParameterizedTest(name="test_get_crystal {index} -> {arguments}")
	@NullAndEmptySource
	@ValueSource(strings = {"non-existent-crystal"})
	public void test_get_crystal(String crystal) {
		assertThrows(IllegalArgumentException.class, () -> {
			Crystal_Struct cs = Xraylib.Crystal_GetCrystal(crystal);
		});
	}

	@Test
	public void test_bragg_angle() {

		double angle = cs.Bragg_angle(10.0, 1, 1, 1);
		assertEquals(angle, 0.3057795845795849, 1E-6);

		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
            		double angle2 = cs.Bragg_angle(-10.0, 1, 1, 1);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_ENERGY);

		exc = assertThrows(IllegalArgumentException.class, () -> {
            		double angle2 = cs.Bragg_angle(10.0, 0, 0, 0);
		});
		assertEquals(exc.getMessage(), Xraylib.INVALID_MILLER);
	}

	@Test
	public void test_Q_scattering_amplitude() {
		double tmp = cs.Q_scattering_amplitude(10.0, 1, 1, 1, Math.PI/4.0);
		assertEquals(tmp, 0.19184445408324474, 1E-6);

		tmp = cs.Q_scattering_amplitude(10.0, 0, 0, 0, Math.PI/4.0);
		assertEquals(tmp, 0.0);

		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
            		double tmp2 = cs.Q_scattering_amplitude(-10.0, 1, 1, 1, Math.PI/4.0);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_ENERGY);
	}

	@Test
	public void test_atomic_factors() {
        	double[] factors = Xraylib.Atomic_Factors(26, 10.0, 1.0, 10.0);
		assertArrayEquals(factors, new double[]{65.15, -0.22193271025027966, 22.420270655080493}, 1E-6);
			
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
            		double[] factors2 = Xraylib.Atomic_Factors(-1, 10.0, 1.0, 10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
            		double[] factors2 = Xraylib.Atomic_Factors(26, -10.0, 1.0, 10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_ENERGY);

		exc = assertThrows(IllegalArgumentException.class, () -> {
            		double[] factors2 = Xraylib.Atomic_Factors(26, 10.0, -1.0, 10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_Q);

		exc = assertThrows(IllegalArgumentException.class, () -> {
            		double[] factors2 = Xraylib.Atomic_Factors(26, 10.0, 1.0, -10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_DEBYE_FACTOR);
	}

	@Test
	public void test_unit_cell_volume() {
		double tmp = cs.Crystal_UnitCellVolume();
		assertEquals(tmp, 45.376673902751, 1E-5);
	}

	@Test
	public void test_dspacing() {
		double tmp = cs.Crystal_dSpacing(1, 1, 1);
		assertEquals(tmp, 2.0592870875248344, 1E-6);

		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
            		double tmp2 = cs.Crystal_dSpacing(0, 0, 0);
		});
		assertEquals(exc.getMessage(), Xraylib.INVALID_MILLER);
	}

	/* TODO: Test Crystal_F_H_StructureFactor and Crystal_F_H_StructureFactor_Partial */
}
