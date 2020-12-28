tests = [
	'atomiclevelwidth',
	'atomicweight',
	'auger',
	'compoundparser',
	'comptonprofiles',
	'coskron',
	'cross_sections',
	'crystal_diffraction',
	'cs_barns',
	'cs_cp',
	'cs_line',
	'densities',
	'edges',
	'fi',
	'fii',
	'fluor_lines',
	'fluor_yield',
	'jump',
	'kissel_pe',
	'polarized',
	'radrate',
	'refractive_indices',
	'scattering',
	'nist-compounds',
	'radionuclides',
	'error',
	'version',
]

foreach _test : tests
  _test_exec = executable(_test, files('test-' + _test + '.c'), c_args: core_c_args, dependencies: [xraylib_lib_dep, ])
  test(_test, _test_exec, timeout: 30)
endforeach