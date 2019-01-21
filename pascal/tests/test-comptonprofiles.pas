program test_comptonprofiles;

{$APPTYPE CONSOLE}
{$mode objfpc}
{$h+}

uses xraylib, xrltest, Classes, SysUtils, fpcunit, testreport, testregistry;

type
	TestComptonProfiles = class(TTestCase)
	private
		procedure _Test_bad_Z_0;
		procedure _Test_bad_Z_103;
		procedure _Test_bad_pz;
		procedure _Test_bad_Z_0_partial;
		procedure _Test_bad_Z_103_partial;
		procedure _Test_bad_pz_partial;
		procedure _Test_bad_shell_low_partial;
		procedure _Test_bad_shell_high_partial;
	published
		procedure Test_pz_0;
		procedure Test_pz_100;
		procedure Test_pz_50;
		procedure Test_bad_input;
	end;


procedure TestComptonProfiles.Test_pz_0;
var
	profile, profile1, profile2: double;
begin
	profile := ComptonProfile(26, 0.0);
	AssertEquals(7.060, profile, 1E-6);
	profile := ComptonProfile_Partial(26, N1_SHELL, 0.0);
	AssertEquals(1.550, profile, 1E-6);
	profile1 := ComptonProfile_Partial(26, L2_SHELL, 0.0);
	profile2 := ComptonProfile_Partial(26, L3_SHELL, 0.0);
	AssertEquals(profile1, profile2, 1E-6);
end;

procedure TestComptonProfiles.Test_pz_100;
var
	profile, profile1, profile2: double;
begin
	profile := ComptonProfile(26, 100.0);
	AssertEquals(1.800E-5, profile, 1E-8);
	profile := ComptonProfile_Partial(26, N1_SHELL, 100.0);
	AssertEquals(5.100E-09, profile, 1E-12);
	profile1 := ComptonProfile_Partial(26, L2_SHELL, 100.0);
	profile2 := ComptonProfile_Partial(26, L3_SHELL, 100.0);
	AssertEquals(profile1, profile2, 1E-10);
	AssertEquals(profile1, 1.100E-8, 1E-10);
end;

procedure TestComptonProfiles.Test_pz_50;
var
	profile, profile1, profile2: double;
begin
	profile := ComptonProfile(26, 50.0);
	AssertEquals(0.0006843950273082384, profile, 1E-8);
	profile := ComptonProfile_Partial(26, N1_SHELL, 50.0);
	AssertEquals(2.4322755767709126e-07, profile, 1E-10);
	profile1 := ComptonProfile_Partial(26, L2_SHELL, 50.0);
	profile2 := ComptonProfile_Partial(26, L3_SHELL, 50.0);
	AssertEquals(profile1, profile2, 1E-10);
	AssertEquals(profile1, 2.026953933016568e-06, 1E-10);
end;

procedure TestComptonProfiles.Test_bad_input;
begin
	AssertException(EArgumentException, @_Test_bad_Z_0);
	ComptonProfile(102, 0.0);
	AssertException(EArgumentException, @_Test_bad_Z_103);
	AssertException(EArgumentException, @_Test_bad_pz);

	AssertException(EArgumentException, @_Test_bad_Z_0_partial);
	ComptonProfile_Partial(102, K_SHELL, 0.0);
	AssertException(EArgumentException, @_Test_bad_Z_103_partial);
	AssertException(EArgumentException, @_Test_bad_pz_partial);
	AssertException(EArgumentException, @_Test_bad_shell_low_partial);
	AssertException(EArgumentException, @_Test_bad_shell_high_partial);
end;

procedure TestComptonProfiles._Test_bad_Z_0;
begin
	ComptonProfile(0, 0.0);
end;

procedure TestComptonProfiles._Test_bad_Z_103;
begin
	ComptonProfile(103, 0.0);
end;

procedure TestComptonProfiles._Test_bad_pz;
begin
	ComptonProfile(26, -1.0);
end;

procedure TestComptonProfiles._Test_bad_Z_0_partial;
begin
	ComptonProfile_Partial(0, K_SHELL, 0.0);
end;

procedure TestComptonProfiles._Test_bad_Z_103_partial;
begin
	ComptonProfile_Partial(103, K_SHELL, 0.0);
end;

procedure TestComptonProfiles._Test_bad_pz_partial;
begin
	ComptonProfile_Partial(26, K_SHELL, -1.0);
end;

procedure TestComptonProfiles._Test_bad_shell_low_partial;
begin
	ComptonProfile_Partial(26, -1, 0.0);
end;

procedure TestComptonProfiles._Test_bad_shell_high_partial;
begin
	ComptonProfile_Partial(26, N2_SHELL, 0.0);
end;

var
	App: TestRunner;
begin
	RegisterTest(TestComptonProfiles);
	App := TestRunner.Create(nil);
	App.Initialize;
	App.Run;
	App.Free;
end.

