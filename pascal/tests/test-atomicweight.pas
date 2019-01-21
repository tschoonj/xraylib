program test_atomicweight;

{$APPTYPE CONSOLE}
{$mode objfpc}
{$h+}

uses xraylib, xrltest, Classes, SysUtils, fpcunit, testreport, testregistry;

type
	TestAtomicWeight = class(TTestCase)
	private
		procedure _Test_bad_Z;
	published
		procedure Test_Fe;
		procedure Test_U;
		procedure Test_bad_Z;
	end;

procedure TestAtomicWeight.Test_Fe;
var
	weight: double;
begin
	weight:= AtomicWeight(26);
	AssertEquals(55.850, weight, 1E-6);
end;

procedure TestAtomicWeight.Test_U;
var
	weight: double;
begin
	weight := AtomicWeight(92);
	AssertEquals(238.070, weight, 1E-6);
end;

procedure TestAtomicWeight._Test_bad_Z;
var
	weight: double;
begin
	weight := AtomicWeight(185);
end;

procedure TestAtomicWeight.Test_bad_Z;
begin
	AssertException(EArgumentException, @_Test_bad_Z);
end;

var
	App: TestRunner;
begin
	RegisterTest(TestAtomicWeight);
	App := TestRunner.Create(nil);
	App.Initialize;
	App.Run;
	App.Free;
end.

