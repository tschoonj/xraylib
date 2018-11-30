require "xraylib"
require "test/unit"

class TestAtomicLevelWidth < Test::Unit::TestCase
	def test_Fe_K
		width = Xraylib.AtomicLevelWidth(26, Xraylib::K_SHELL)
		assert_in_delta(width, 1.19E-3, 1E-6)
	end

	def test_U_N7
		width = Xraylib.AtomicLevelWidth(92, Xraylib::N7_SHELL)
		assert_in_delta(width, 0.31E-3, 1E-6)
	end

	def test_bad_Z
		assert_raise(ArgumentError) {
			Xraylib.AtomicLevelWidth(185, Xraylib::K_SHELL)
		}
		assert_raise(ArgumentError) {
			Xraylib.AtomicLevelWidth()
		}
		assert_raise(TypeError) {
			Xraylib.AtomicLevelWidth(nil, Xraylib::K_SHELL)
		}
		assert_raise(TypeError) {
			Xraylib.AtomicLevelWidth("hwoefhhfowfhwfh", Xraylib::K_SHELL)
		}
		assert_raise(TypeError) {
			Xraylib.AtomicLevelWidth("26", Xraylib::K_SHELL)
		}
	end

	def test_bad_shell
		assert_raise(ArgumentError) {
			Xraylib.AtomicLevelWidth(26, -5)
		}
	end

	def test_invalid_Z
		assert_raise(ArgumentError) {
			Xraylib.AtomicLevelWidth(26, Xraylib::N3_SHELL)
		}
	end
end

