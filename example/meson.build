windows_installer_example_files = files(
    'xrlexample1.c',
    'xrlexample6.cpp',
    'xrlexample8.cs',
    'xrlexample14.pas',
)

# xrlexample1 -> C
xrlexample1_exec = executable('xrlexample1', files('xrlexample1.c'), c_args: core_c_args, dependencies: [xraylib_lib_dep, ])
test('xrlexample1', xrlexample1_exec, timeout: 30)

# xrlexample3 -> Fortran
if fortran_enabled
  xrlexample3_exec = executable('xrlexample3', files('xrlexample3.f90'), dependencies: [xraylib_fortran_lib_dep, ])
  test('xrlexample3', xrlexample3_exec, timeout: 30)
endif

# xrlexample5 -> Python (swig)
if python_enabled
  test('xrlexample5',
    python,
    args : files('xrlexample5.py'),
    env : python_test_env,
    timeout: 10,
    depends: [xraylib_ext]
  )
endif

# xrlexample6.cpp -> C++
if cplusplus_enabled
  xrlexample6_exec = executable('xrlexample6', files('xrlexample6.cpp'),  dependencies: [xraylib_lib_dep, ], include_directories: ['../cplusplus'])
  test('xrlexample6', xrlexample6_exec, timeout: 30)
endif

# xrlexample13 -> Python (cython)
if python_numpy_enabled
  test('xrlexample13',
    python,
    args : files('xrlexample13.py'),
    env : python_test_env,
    timeout: 10,
    depends: [xraylib_np_ext]
  )
endif