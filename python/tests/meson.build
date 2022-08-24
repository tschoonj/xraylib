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
    # see https://github.com/xhochy/python-feedstock/blob/6efc12d0479e958a138bfaf3d1eb6b47a10d5f68/recipe/patches/0014-Add-CondaEcosystemModifyDllSearchPath.patch
    test_env.prepend('CONDA_DLL_SEARCH_MODIFICATION_ENABLE', '1')
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