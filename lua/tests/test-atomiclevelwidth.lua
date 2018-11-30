require("xraylib")
luaunit = require("luaunit")

TestAtomicLevelWidth = {}
	function TestAtomicLevelWidth:test_Fe_K()
		local width = xraylib.AtomicLevelWidth(26, xraylib.K_SHELL)
		luaunit.assertAlmostEquals(width, 1.19E-3)
	end

	function TestAtomicLevelWidth:test_U_N7()
		local width = xraylib.AtomicLevelWidth(92, xraylib.N7_SHELL)
		luaunit.assertAlmostEquals(width, 0.31E-3)
	end

	function TestAtomicLevelWidth:test_bad_Z()
		luaunit.assertError(xraylib.AtomicLevelWidth, 185, xraylib.K_SHELL)
	end

	function TestAtomicLevelWidth:test_bad_shell()
		luaunit.assertError(xraylib.AtomicLevelWidth, 26, -5)
	end

	function TestAtomicLevelWidth:test_invalid_shell()
		luaunit.assertError(xraylib.AtomicLevelWidth, 26, xraylib.N3_SHELL)
	end

os.exit(luaunit.LuaUnit.run('-v', '-o', 'tap'))
