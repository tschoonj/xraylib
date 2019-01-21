program test_auger;

{$APPTYPE CONSOLE}
{$mode objfpc}
{$h+}

uses xraylib, xrltest, Classes, SysUtils, fpcunit, testreport, testregistry;

type
	TestAugerRate = class(TTestCase)
	private
		procedure _Test_bad_Z;
		procedure _Test_bad_trans;
		procedure _Test_invalid_trans;
	published
		procedure Test_Pb_K_L3M5;
		procedure Test_Pb_L3_M4N7;
		procedure Test_bad_Z;
		procedure Test_bad_trans;
		procedure Test_invalid_trans;
	end;
	
type
	TestAugerYield = class(TTestCase)
	private
		procedure _Test_bad_Z;
		procedure _Test_bad_shell;
	published
		procedure Test_Pb_K;
		procedure Test_Pb_M3;
		procedure Test_bad_Z;
		procedure Test_bad_shell;
	end;
	
procedure TestAugerRate.Test_Pb_K_L3M5;
var
	rate: double;
begin
	rate := AugerRate(82, K_L3M5_AUGER);
	AssertEquals(0.004573193387, rate, 1E-6);
end;

procedure TestAugerRate.Test_Pb_L3_M4N7;
var
	rate: double;
begin
	rate := AugerRate(82, L3_M4N7_AUGER);
	AssertEquals(0.0024327572005, rate, 1E-6);
end;

procedure TestAugerRate._Test_bad_Z;
var
	rate: double;
begin
	rate := AugerRate(-35, L3_M4N7_AUGER);
end;

procedure TestAugerRate._Test_bad_trans;
var
	rate: double;
begin
	rate := AugerRate(82, M4_M5Q3_AUGER + 1);
end;

procedure TestAugerRate._Test_invalid_trans;
var
	rate: double;
begin
	rate := AugerRate(62, L3_M4N7_AUGER);
end;

procedure TestAugerRate.Test_bad_Z;
begin
	AssertException(EArgumentException, @_Test_bad_Z);
end;

procedure TestAugerRate.Test_bad_trans;
begin
	AssertException(EArgumentException, @_Test_bad_trans);
end;

procedure TestAugerRate.Test_invalid_trans;
begin
	AssertException(EArgumentException, @_Test_invalid_trans);
end;

procedure TestAugerYield.Test_Pb_K;
var
	yield: double;
begin
	yield := AugerYield(82, K_SHELL);
	AssertEquals(1.0 - FluorYield(82, K_SHELL), yield, 1E-6);
end;

procedure TestAugerYield.Test_Pb_M3;
var
	yield: double;
begin
	yield := AugerYield(82, M3_SHELL);
	AssertEquals(0.1719525, yield, 1E-6);
end;

procedure TestAugerYield._Test_bad_Z;
var
	yield: double;
begin
	yield := AugerYield(-35, K_SHELL);
end;

procedure TestAugerYield._Test_bad_shell;
var
	yield: double;
begin
	yield := AugerYield(82, N2_SHELL);
end;

procedure TestAugerYield.Test_bad_Z;
begin
	AssertException(EArgumentException, @_Test_bad_Z);
end;

procedure TestAugerYield.Test_bad_shell;
begin
	AssertException(EArgumentException, @_Test_bad_shell);
end;

var
	App: TestRunner;
begin
	RegisterTest(TestAugerRate);
	RegisterTest(TestAugerYield);
	App := TestRunner.Create(nil);
	App.Initialize;
	App.Run;
	App.Free;
end.

