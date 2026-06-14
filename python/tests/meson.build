python_tests = [
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

python_numpy_tests = []

python_test_env = environment()
python_test_env.prepend('PYTHONPATH', pydir)
# Suppress OpenMP runtime conflict abort (e.g., when numpy and xraylib_np link different libomp variants on macOS)
python_test_env.set('KMP_DUPLICATE_LIB_OK', 'TRUE')
if host_system == 'windows'
    python_test_env.prepend('PATH', src_build_dir)
    python_test_env.append('EXTRA_DLL_SEARCH_PATHS', src_build_dir)
endif

if python_enabled
  foreach _test : python_tests
    test('python-' + _test,
      python,
      args : files('test-' + _test + '.py'),
      env : python_test_env,
      timeout: 10,
      depends: [xraylib_ext]
    )
  endforeach
endif

if python_numpy_enabled
  foreach _test : python_numpy_tests
    test('python-' + _test,
      python,
      args : files('test-' + _test + '.py'),
      env : python_test_env,
      timeout: 10,
      depends: [xraylib_np_ext]
    )
  endforeach
endif