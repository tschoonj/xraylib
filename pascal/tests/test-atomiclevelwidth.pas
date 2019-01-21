program test_atomiclevelwidth;

{$APPTYPE CONSOLE}
{$mode objfpc}
{$h+}

uses xraylib, xrltest, Classes, SysUtils, fpcunit, testreport, testregistry;

type
	TestAtomicLevelWidth = class(TTestCase)
	private
		procedure _Test_bad_Z;
		procedure _Test_bad_shell;
		procedure _Test_invalid_shell;
	published
		procedure Test_Fe_K;
		procedure Test_U_N7;
		procedure Test_bad_Z;
		procedure Test_bad_shell;
		procedure Test_invalid_shell;
	end;

procedure TestAtomicLevelWidth.Test_Fe_K;
var
	width: double;
begin
	width := AtomicLevelWidth(26, K_SHELL);
	AssertEquals(1.19E-3, width, 1E-6);
end;

procedure TestAtomicLevelWidth.Test_U_N7;
var
	width: double;
begin
	width := AtomicLevelWidth(92, N7_SHELL);
	AssertEquals(0.31E-3, width, 1E-6);
end;

procedure TestAtomicLevelWidth._Test_bad_Z;
var
	width: double;
begin
	width := AtomicLevelWidth(185, N7_SHELL);
end;

procedure TestAtomicLevelWidth._Test_bad_shell;
var
	width: double;
begin
	width := AtomicLevelWidth(185, -5);
end;

procedure TestAtomicLevelWidth._Test_invalid_shell;
var
	width: double;
begin
	width := AtomicLevelWidth(23, N3_SHELL);
end;

procedure TestAtomicLevelWidth.Test_bad_Z;
begin
	AssertException(EArgumentException, @_Test_bad_Z);
end;

procedure TestAtomicLevelWidth.Test_bad_shell;
begin
	AssertException(EArgumentException, @_Test_bad_shell);
end;

procedure TestAtomicLevelWidth.Test_invalid_shell;
begin
	AssertException(EArgumentException, @_Test_invalid_shell);
end;

var
	App: TestRunner;
begin
	RegisterTest(TestAtomicLevelWidth);
	App := TestRunner.Create(nil);
	App.Initialize;
	App.Run;
	App.Free;
end.
