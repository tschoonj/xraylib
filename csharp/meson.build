windows = import('windows')

win_res = windows.compile_resources('app.rc', depend_files: files('app.ico', 'resource.h'))

libxrl_cs_sources = files(
    'AssemblyInfo.cpp',
    'Compound.h',
    'Diffraction.h',
    'Element.h',
    'Errors.h',
    'Radionuclides.h',
    'Stdafx.cpp',
    'Stdafx.h',
    'XrayLib.NET.cpp',
    'XrayLib.NET.h',
)

xraylib_cs_lib = shared_library(
    'XrayLib.NET',
    win_res,
    libxrl_cs_sources,
    dependencies: xraylib_lib_dep,
    cpp_args: ['/clr', '/Zc:twoPhase-',],  
    override_options: ['cpp_eh=none'],
)

xraylib_cs_dep = declare_dependency(
    link_with: xraylib_cs_lib,
)

csharp_source_dir = meson.current_source_dir()
csharp_build_dir = meson.current_build_dir()

# Note: to avoid the executable not finding the assembly at run-time,
# ensure they are created in the same folder...
_test_exec = executable(
    'Program', ['Program.cs', 'AssemblyInfo.cs'],
    dependencies: [xraylib_cs_dep],
    cs_args: ['-r:System.Numerics.dll',])
test('cs-Program', _test_exec,)