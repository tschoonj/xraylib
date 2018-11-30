use Test::More;
use xraylib;
use xrltest;

ok(xrltest::almost_equal(xraylib::AugerRate(82, $xraylib::K_L3M5_AUGER), 0.004573193387), "TestAugerRate::test_Pb_K_L3M5");
ok(xrltest::almost_equal(xraylib::AugerRate(82, $xraylib::L3_M4N7_AUGER), 0.0024327572005), "TestAugerRate::test_Pb_L3_M4N7");
eval {
	xraylib::AugerRate(-35, $xraylib::L3_M4N7_AUGER);
};
like($@, qr/^ValueError/ , "TestAugerRate::test_bad_Z");
eval {
	xraylib::AugerRate(82, $xraylib::M4_M5Q3_AUGER + 1);
};
like($@, qr/^ValueError/ , "TestAugerRate::test_bad_trans");
eval {
	xraylib::AugerRate(62, $xraylib::L3_M4N7_AUGER);
};
like($@, qr/^ValueError/ , "TestAugerRate::test_invalid_trans");

ok(xrltest::almost_equal(xraylib::AugerYield(82, $xraylib::K_SHELL), 1.0 - xraylib::FluorYield(82, $xraylib::K_SHELL)), "TestAugerYield::test_Pb_K");
ok(xrltest::almost_equal(xraylib::AugerYield(82, $xraylib::M3_SHELL), 0.1719525), "TestAugerYield::test_Pb_M3");
eval {
	xraylib::AugerYield(-35, $xraylib::K_SHELL);
};
like($@, qr/^ValueError/ , "TestAugerYield::test_bad_Z");
eval {
	xraylib::AugerYield(82, $xraylib::N2_SHELL);
};
like($@, qr/^ValueError/ , "TestAugerYield::test_bad_shell");


done_testing();

