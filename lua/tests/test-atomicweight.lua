require("xraylib")
luaunit = require("luaunit")

TestAtomicWeight = {}
	function TestAtomicWeight:test_Fe()
		local weight = xraylib.AtomicWeight(26)
		luaunit.assertAlmostEquals(weight, 55.850)
	end

	function TestAtomicWeight:test_U()
		local weight = xraylib.AtomicWeight(92)
		luaunit.assertAlmostEquals(weight, 238.070)
	end

	function TestAtomicWeight:test_bad_Z()
		luaunit.assertError(xraylib.AtomicWeight, 185)
	end

os.exit(luaunit.LuaUnit.run('-v', '-o', 'tap'))

