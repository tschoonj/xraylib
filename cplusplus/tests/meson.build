tests = [
	'atomiclevelwidth',
	'atomicweight',
	'compoundparser',
	'crystal_diffraction',
	'nist-compounds',
	'radionuclides',
	'refractive_indices',
        'xrl_functions',
]

foreach _test : tests
  _test_exec = executable(_test, files('test-' + _test + '.cpp'),  dependencies: [xraylib_lib_dep, ], include_directories: ['..'])
  test('c++-' + _test, _test_exec, timeout: 30)
endforeach
