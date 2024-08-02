cplusplus_source_files = files('xraylib++.h')

install_headers(cplusplus_source_files, subdir: 'xraylib')

cplusplus_source_dir = meson.current_source_dir()

if add_languages('cpp', required: false, native: false)
    subdir('tests')
else
    warning('The C++ tests will not get built and run')
endif