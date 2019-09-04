import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.Arguments;
import static org.junit.jupiter.params.provider.Arguments.arguments;
 
import com.github.tschoonj.xraylib.Xraylib;

public class TestCrossSections {

	@FunctionalInterface
	private static interface CrossSectionWrapper {
		public double execute(int Z, double energy);
	}

	static Stream<Arguments> goodValuesProvider() {
		return Stream.of(
			arguments((CrossSectionWrapper) Xraylib::CS_Photo, 11.451033638148562),
			arguments((CrossSectionWrapper) Xraylib::CS_Compt, 0.11785269096475783),
			arguments((CrossSectionWrapper) Xraylib::CS_Rayl, 0.39841164641058013),
			arguments((CrossSectionWrapper) Xraylib::CS_Total, 11.451033638148562 + 0.11785269096475783 + 0.39841164641058013),
			arguments((CrossSectionWrapper) Xraylib::CS_Energy, 11.420221747941419)
		);
	}

	@ParameterizedTest(name="test_good_values {index} -> {0} {1}")
	@MethodSource("goodValuesProvider")
	public void test_good_values(CrossSectionWrapper wrapper, double expected) {
		double cs = wrapper.execute(10, 10.0);
		assertEquals(cs, expected, 1E-4);
	}

	static Stream<CrossSectionWrapper> badValuesProvider() {
		return Stream.of(
			Xraylib::CS_Photo,
			Xraylib::CS_Compt,
			Xraylib::CS_Rayl,
			Xraylib::CS_Total,
			Xraylib::CS_Energy
		);
	}

	@ParameterizedTest(name="bad_good_values {index} -> {0} {1}")
	@MethodSource("badValuesProvider")
	public void test_bad_values(CrossSectionWrapper wrapper) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = wrapper.execute(-1, 10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = wrapper.execute(Xraylib.ZMAX, 10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = wrapper.execute(26, 0.0);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_ENERGY);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = wrapper.execute(26, -1.0);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_ENERGY);
	}
}
