require "xraylib"
require "test/unit"

class TestAtomicWeight < Test::Unit::TestCase
	def test_Fe
		weight = Xraylib.AtomicWeight(26)
		assert_in_delta(weight, 55.850, 1E-6)
	end

	def test_U
		weight = Xraylib.AtomicWeight(92)
		assert_in_delta(weight, 238.070, 1E-6)
	end

	def test_bad_Z
		assert_raise(ArgumentError) {
			Xraylib.AtomicWeight(185)
		}
		assert_raise(ArgumentError) {
			Xraylib.AtomicWeight()
		}
		assert_raise(TypeError) {
			Xraylib.AtomicWeight(nil)
		}
		assert_raise(TypeError) {
			Xraylib.AtomicWeight("hwoefhhfowfhwfh")
		}
		assert_raise(TypeError) {
			Xraylib.AtomicWeight("26")
		}
		assert_raise(ArgumentError) {
			Xraylib.AtomicWeight(26, -5)
		}
	end
end


