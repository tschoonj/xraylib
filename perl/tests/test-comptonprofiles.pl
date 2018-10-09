use strict;
use Test::More;
use xraylib;
use xrltest;

ok(xrltest::almost_equal(xraylib::ComptonProfile(26, 0.0), 7.060), "TestComptonProfiles::test_pz_0_full");
ok(xrltest::almost_equal(xraylib::ComptonProfile_Partial(26, $xraylib::N1_SHELL, 0.0), 1.550), "TestComptonProfiles::test_pz_0_partial");
my $profile1 = xraylib::ComptonProfile_Partial(26, $xraylib::L2_SHELL, 0.0);
my $profile2 = xraylib::ComptonProfile_Partial(26, $xraylib::L3_SHELL, 0.0);
ok(xrltest::almost_equal($profile1, $profile2), "TestComptonProfiles::test_pz0_L2_L3_part1");
ok(xrltest::almost_equal($profile1, 0.065), "TestComptonProfiles::test_pz0_L2_L3_part2");

ok(xrltest::almost_equal(xraylib::ComptonProfile(26, 100.0), 1.8E-5, 1E-8), "TestComptonProfiles::test_pz_100_full");
ok(xrltest::almost_equal(xraylib::ComptonProfile_Partial(26, $xraylib::N1_SHELL, 100.0), 5.1E-9, 1E-12), "TestComptonProfiles::test_pz_100_partial");
my $profile1 = xraylib::ComptonProfile_Partial(26, $xraylib::L2_SHELL, 100.0);
my $profile2 = xraylib::ComptonProfile_Partial(26, $xraylib::L3_SHELL, 100.0);
ok(xrltest::almost_equal($profile1, $profile2, 1E-10), "TestComptonProfiles::test_pz100_L2_L3_part1");
ok(xrltest::almost_equal($profile1, 1.1E-8, 1E-10), "TestComptonProfiles::test_pz100_L2_L3_part2");

ok(xrltest::almost_equal(xraylib::ComptonProfile(26, 50.0), 0.0006843950273082384, 1E-8), "TestComptonProfiles::test_pz_50_full");
ok(xrltest::almost_equal(xraylib::ComptonProfile_Partial(26, $xraylib::N1_SHELL, 50.0), 2.4322755767709126e-07, 1E-10), "TestComptonProfiles::test_pz_50_partial");
my $profile1 = xraylib::ComptonProfile_Partial(26, $xraylib::L2_SHELL, 50.0);
my $profile2 = xraylib::ComptonProfile_Partial(26, $xraylib::L3_SHELL, 50.0);
ok(xrltest::almost_equal($profile1, $profile2, 1E-10), "TestComptonProfiles::test_pz50_L2_L3_part1");
ok(xrltest::almost_equal($profile1, 2.026953933016568e-06, 1E-10), "TestComptonProfiles::test_pz50_L2_L3_part2");

eval {
	xraylib::ComptonProfile(0, 0.0);
};
like($@, qr/^ValueError/ , "TestComptonProfiles::test_bad_input_full_lowZ");
xraylib::ComptonProfile(102, 0.0);
eval {
	xraylib::ComptonProfile(103, 0.0);
};
like($@, qr/^ValueError/ , "TestComptonProfiles::test_bad_input_full_highZ");
eval {
	xraylib::ComptonProfile(26, -1.0);
};
like($@, qr/^ValueError/ , "TestComptonProfiles::test_bad_input_full_negative_pz");
eval {
	xraylib::ComptonProfile_Partial(0, $xraylib::K_SHELL, 0.0);
};
like($@, qr/^ValueError/ , "TestComptonProfiles::test_bad_input_partial_lowZ");
xraylib::ComptonProfile_Partial(102, $xraylib::K_SHELL, 0.0);
eval {
	xraylib::ComptonProfile_Partial(103, $xraylib::K_SHELL, 0.0);
};
like($@, qr/^ValueError/ , "TestComptonProfiles::test_bad_input_partial_highZ");
eval {
	xraylib::ComptonProfile_Partial(26, $xraylib::K_SHELL, -1.0);
};
like($@, qr/^ValueError/ , "TestComptonProfiles::test_bad_input_partial_negative_pz");
eval {
	xraylib::ComptonProfile_Partial(26, -1, 0.0);
};
like($@, qr/^ValueError/ , "TestComptonProfiles::test_bad_input_partial_low_shell");
eval {
	xraylib::ComptonProfile_Partial(26, $xraylib::N2_SHELL, 0.0);
};
like($@, qr/^ValueError/ , "TestComptonProfiles::test_bad_input_partial_high_shell");
eval {
	xraylib::ComptonProfile_Partial(26, $xraylib::N2_SHELL);
};
like($@, qr/^RuntimeError/ , "TestComptonProfiles::test_bad_input_partial_missing_pz");
eval {
	xraylib::ComptonProfile_Partial(26, $xraylib::N2_SHELL, "jpjjpgjgjgp");
};
like($@, qr/^TypeError/ , "TestComptonProfiles::test_bad_input_partial_string_pz");





done_testing();
