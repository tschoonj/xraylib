test_lib = static_library(
    'unittest',
    files('libtest.f90'),
)

test_lib_dep = declare_dependency(
    link_with: test_lib,
)

tests = [
	'atomiclevelwidth',
	'atomicweight',
	'auger',
	'compoundparser',
	'comptonprofiles',
	'crystal_diffraction',
	'refractive_indices',
	'nist-compounds',
	'radionuclides',
]

foreach _test : tests
  _test_exec = executable(_test, files('test-' + _test + '.f90'), dependencies: [xraylib_fortran_lib_dep, test_lib_dep])
  test('fortran-' + _test, _test_exec, timeout: 30)
endforeach
