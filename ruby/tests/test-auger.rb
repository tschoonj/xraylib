require "xraylib"
require "test/unit"

class TestAugerRate < Test::Unit::TestCase
	def test_Pb_K_L3M5
		rate = Xraylib.AugerRate(82, Xraylib::K_L3M5_AUGER)
		assert_in_delta(rate, 0.004573193387, 1E-6)
	end

	def test_Pb_L3_M4N7
		rate = Xraylib.AugerRate(82, Xraylib::L3_M4N7_AUGER)
		assert_in_delta(rate, 0.0024327572005, 1E-6)
	end

	def test_bad_Z
		assert_raise(ArgumentError) {
			Xraylib.AugerRate(-35, Xraylib::L3_M4N7_AUGER)
		}
		assert_raise(ArgumentError) {
			Xraylib.AugerRate(180, Xraylib::L3_M4N7_AUGER)
		}
		assert_raise(ArgumentError) {
			Xraylib.AugerRate()
		}
		assert_raise(ArgumentError) {
			Xraylib.AugerRate(nil)
		}
		assert_raise(TypeError) {
			Xraylib.AugerRate("hwoefhhfowfhwfh", Xraylib::L3_M4N7_AUGER)
		}
		assert_raise(TypeError) {
			Xraylib.AugerRate("26", Xraylib::K_L3M5_AUGER)
		}
	end

	def test_bad_trans
		assert_raise(ArgumentError) {
			Xraylib.AugerRate(82, Xraylib::M4_M5Q3_AUGER + 1)
		}
	end

	def test_invalid_trans
		assert_raise(ArgumentError) {
			Xraylib.AugerRate(62, Xraylib::L3_M4N7_AUGER)
		}
	end
end

class TestAugerYield < Test::Unit::TestCase
	def test_Pb_K
		ayield = Xraylib.AugerYield(82, Xraylib::K_SHELL)
		assert_in_delta(ayield, 1.0 - Xraylib.FluorYield(82, Xraylib::K_SHELL), 1E-6)
	end

	def test_Pb_M3
		ayield = Xraylib.AugerYield(82, Xraylib::M3_SHELL)
		assert_in_delta(ayield, 0.1719525, 1E-6)
	end

	def test_bad_Z
		assert_raise(ArgumentError) {
			Xraylib.AugerYield(-35, Xraylib::K_SHELL)
		}
	end

	def test_bad_shell
		assert_raise(ArgumentError) {
			Xraylib.AugerYield(82, Xraylib::N2_SHELL)
		}
	end
end
