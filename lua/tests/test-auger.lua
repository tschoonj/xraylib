require("xraylib")
luaunit = require("luaunit")

TestAugerRate = {}
	function TestAugerRate:test_Pb_K_L3M5()
		local rate = xraylib.AugerRate(82, xraylib.K_L3M5_AUGER)
		luaunit.assertAlmostEquals(rate, 0.004573193387)
	end

	function TestAugerRate:test_Pb_L3_M4N7()
		local rate = xraylib.AugerRate(82, xraylib.L3_M4N7_AUGER)
		luaunit.assertAlmostEquals(rate, 0.0024327572005)
	end

	function TestAugerRate:test_bad_Z()
		luaunit.assertError(xraylib.AugerRate, -35, xraylib.L3_M4N7_AUGER)
	end

	function TestAugerRate:test_bad_trans()
		luaunit.assertError(xraylib.AugerRate, 82, xraylib.M4_M5Q3_AUGER + 1)
	end

	function TestAugerRate:test_invalid_trans()
		luaunit.assertError(xraylib.AugerRate, 62, xraylib.L3_M4N7_AUGER)
	end

TestAugerYield = {}
	function TestAugerYield:test_Pb_K()
		local ayield = xraylib.AugerYield(82, xraylib.K_SHELL)
		luaunit.assertAlmostEquals(ayield, 1.0 - xraylib.FluorYield(82, xraylib.K_SHELL))
	end

	function TestAugerYield:test_Pb_M3()
		local ayield = xraylib.AugerYield(82, xraylib.M3_SHELL)
		luaunit.assertAlmostEquals(ayield, 0.1719525)
	end

	function TestAugerYield:test_bad_Z()
		luaunit.assertError(xraylib.AugerYield, -35, xraylib.K_SHELL)
	end

	function TestAugerYield:test_bad_shell()
		luaunit.assertError(xraylib.AugerYield, 82, xraylib.N2_SHELL)
	end

os.exit(luaunit.LuaUnit.run('-v', '-o', 'tap'))
