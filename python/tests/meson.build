tests = [
	'atomiclevelwidth',
	'atomicweight',
	'auger',
	'compoundparser',
	'comptonprofiles',
	'crystal_diffraction',
	'nist-compounds',
	'radionuclides',
	'numpy',
]

test_env = environment()
test_env.prepend('PYTHONPATH', pydir)
if host_system == 'windows'
    test_env.prepend('PATH', src_build_dir)
    test_env.append('EXTRA_DLL_SEARCH_PATHS', src_build_dir)
endif

foreach _test : tests
  test('python-' + _test,
    python,
    args : files('test-' + _test + '.py'),
    env : test_env,
    timeout: 10,
    depends: [xraylib_ext]
  )
endforeach