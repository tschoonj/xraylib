use strict;
use Test::More;
use xraylib;
use xrltest;

my $list = xraylib::GetRadioNuclideDataList();
ok(scalar @{$list} eq 10, "TestRadionuclides:test_good::list");
for my $i (0..9) {
	my $rnd = xraylib::GetRadioNuclideDataByIndex($i);
	ok($rnd->{name} eq $list->[$i], "TestRadionuclides:test_good::index");
	$rnd = xraylib::GetRadioNuclideDataByName($list->[$i]);
	ok($rnd->{name} eq $list->[$i], "TestRadionuclides:test_good::name");
}

my $rnd = xraylib::GetRadioNuclideDataByIndex(3);
ok($rnd->{'A'} eq 125, "TestRadionuclides::test_good_nr3::A");
ok($rnd->{'GammaEnergies'} ~~ [35.4919], "TestRadionuclides::test_good_nr3::GammaEnergies");
ok($rnd->{'GammaIntensities'} ~~ [0.0668], "TestRadionuclides::test_good_nr3::GammaIntensities");
ok($rnd->{'N'} eq 72, "TestRadionuclides::test_good_nr3::N");
ok($rnd->{XrayIntensities} ~~ [0.0023, 0.00112, 0.0063, 0.056, 0.035, 0.0042, 0.007, 0.00043, 0.0101, 0.0045, 0.00103, 0.0016, 3.24e-05, 0.406, 0.757, 0.0683, 0.132, 0.00121, 0.0381, 0.0058], "TestRadionuclides::test_good_nr3::XrayIntensities");
ok($rnd->{XrayLines} ~~ [-86, -60, -89, -90, -63, -33, -34, -91, -95, -68, -38, -39, -1, -2, -3, -5, -6, -8, -11, -13], "TestRadionuclides::test_good_nr3::XrayLines");
ok($rnd->{'Z'} eq 53, "TestRadionuclides::test_good_nr3::Z");
ok($rnd->{'Z_xray'} eq 52, "TestRadionuclides::test_good_nr3::Z_xray");
ok($rnd->{'nGammas'} eq 1, "TestRadionuclides::test_good_nr3::nGammas");
ok($rnd->{'nXrays'} eq 20, "TestRadionuclides::test_good_nr3::nXrays");
ok($rnd->{'name'} eq "125I", "TestRadionuclides::test_good_nr3::name");

$rnd = xraylib::GetRadioNuclideDataByName("125I");
ok($rnd->{'A'} eq 125, "TestRadionuclides::test_good_125I::A");
ok($rnd->{'GammaEnergies'} ~~ [35.4919], "TestRadionuclides::test_good_125I::GammaEnergies");
ok($rnd->{'GammaIntensities'} ~~ [0.0668], "TestRadionuclides::test_good_125I::GammaIntensities");
ok($rnd->{'N'} eq 72, "TestRadionuclides::test_good_125I::N");
ok($rnd->{XrayIntensities} ~~ [0.0023, 0.00112, 0.0063, 0.056, 0.035, 0.0042, 0.007, 0.00043, 0.0101, 0.0045, 0.00103, 0.0016, 3.24e-05, 0.406, 0.757, 0.0683, 0.132, 0.00121, 0.0381, 0.0058], "TestRadionuclides::test_good_125I::XrayIntensities");
ok($rnd->{XrayLines} ~~ [-86, -60, -89, -90, -63, -33, -34, -91, -95, -68, -38, -39, -1, -2, -3, -5, -6, -8, -11, -13], "TestRadionuclides::test_good_125I::XrayLines");
ok($rnd->{'Z'} eq 53, "TestRadionuclides::test_good_125I::Z");
ok($rnd->{'Z_xray'} eq 52, "TestRadionuclides::test_good_125I::Z_xray");
ok($rnd->{'nGammas'} eq 1, "TestRadionuclides::test_good_125I::nGammas");
ok($rnd->{'nXrays'} eq 20, "TestRadionuclides::test_good_125I::nXrays");
ok($rnd->{'name'} eq "125I", "TestRadionuclides::test_good_125I::name");


eval {
	xraylib::GetRadioNuclideDataByName('Uu');
};
like($@, qr/^ValueError/ , "TestRadionuclides::test_bad::name1");

eval {
	xraylib::GetRadioNuclideDataByName(undef);
};
like($@, qr/^ValueError/ , "TestRadionuclides::test_bad::name2");
eval {
	xraylib::GetRadioNuclideDataByIndex('Uu');
};
like($@, qr/^TypeError/ , "TestRadionuclides::test_bad::index1");
xraylib::GetRadioNuclideDataByIndex(undef); # undef gets translated to zero apparently!
eval {
	xraylib::GetRadioNuclideDataByIndex(-1);
};
like($@, qr/^ValueError/ , "TestRadionuclides::test_bad::index3");
eval {
	xraylib::GetRadioNuclideDataByIndex(180);
};
like($@, qr/^ValueError/ , "TestRadionuclides::test_bad::index4");


done_testing();

