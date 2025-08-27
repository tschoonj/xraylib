#Copyright (c) 2009-2021, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# this is necessary for CentOS 7
# for compiled modules
%{!?lua_libdir: %global lua_libdir %{_libdir}/lua/%{lua_version}}

# and for Perl, for all distros...
%define perl_vendor_archlib	%(eval "`%__perl -V:installvendorarch`"; echo "$installvendorarch")
%define perl_vendor_autolib	%{perl_vendor_archlib}/auto

Name: xraylib
Version: 4.2.0
Release:	1%{?dist}
Summary: A library for X-ray matter interactions cross sections for X-ray fluorescence applications: core C library
Group:	 Applications/Engineering and Scientific	
License: BSD 
Packager: Tom.Schoonjans <Tom.Schoonjans@me.com>
URL: http://github.com/tschoonj/xraylib
Source: xraylib-%{version}.tar.gz	
BuildRoot:	%(mktemp -ud %{_tmppath}/%{name}-%{version}-%{release}-XXXXXX)
BuildRequires: gcc glibc glibc-headers glibc-devel gcc-gfortran >= 4.3.0 swig lua-devel ruby-devel perl-devel php-devel

%if 0%{?rhel}

%if 0%{?rhel} == 7
# python2-Cython is too old!
%define cython2 /usr/bin/cython3.6
# python 3
BuildRequires: python36-Cython python36-numpy python36-devel python36-setuptools
%define cython3 /usr/bin/cython3.6

%else
# Centos 8
%define cython2 /usr/bin/cython
# python 3
BuildRequires: python3-Cython python3-numpy python3-devel python3-setuptools
%define cython3 /usr/bin/cython

%endif

%else

%if 0%{?fedora}
BuildRequires: python3-Cython python3-numpy python3-devel
%define cython3 /usr/bin/cython
%endif

%endif

%description
Quantitative estimate of elemental composition by spectroscopic and imaging techniques using X-ray fluorescence requires the availability of accurate data of X-ray interaction with matter. Although a wide number of computer codes and data sets are reported in literature, none of them is presented in the form of freely available library functions which can be easily included in software applications for X-ray fluorescence. This work presents a compilation of data sets from different published works and an xraylib interface in the form of callable functions. Although the target applications are on X-ray fluorescence, cross sections of interactions like photoionization, coherent scattering and Compton scattering, as well as form factors and anomalous scattering functions, are also available.

This rpm package provides only the core C library.

%package devel
Summary:A library for X-ray matter interactions cross sections for X-ray fluorescence applications: development package
Requires: glibc-devel glibc-headers pkgconfig %{name}%{?_isa} = %{version}-%{release}

%description devel
Quantitative estimate of elemental composition by spectroscopic and imaging techniques using X-ray fluorescence requires the availability of accurate data of X-ray interaction with matter. Although a wide number of computer codes and data sets are reported in literature, none of them is presented in the form of freely available library functions which can be easily included in software applications for X-ray fluorescence. This work presents a compilation of data sets from different published works and an xraylib interface in the form of callable functions. Although the target applications are on X-ray fluorescence, cross sections of interactions like photoionization, coherent scattering and Compton scattering, as well as form factors and anomalous scattering functions, are also available.

This rpm package provides the necessary libraries, headers etc to start your own xraylib based development.

%package fortran
Summary:A library for X-ray matter interactions cross sections for X-ray fluorescence applications: fortran bindings
Requires: pkgconfig libgfortran >= 4.3.0 gcc-gfortran >= 4.3.0 %{name}%{?_isa} = %{version}-%{release}

%description fortran 
Quantitative estimate of elemental composition by spectroscopic and imaging techniques using X-ray fluorescence requires the availability of accurate data of X-ray interaction with matter. Although a wide number of computer codes and data sets are reported in literature, none of them is presented in the form of freely available library functions which can be easily included in software applications for X-ray fluorescence. This work presents a compilation of data sets from different published works and an xraylib interface in the form of callable functions. Although the target applications are on X-ray fluorescence, cross sections of interactions like photoionization, coherent scattering and Compton scattering, as well as form factors and anomalous scattering functions, are also available.

This rpm package provides the fortran 2003 bindings of xraylib.

%package python
Summary:A library for X-ray matter interactions cross sections for X-ray fluorescence applications: python3 bindings
Requires:  %{name}%{?_isa} = %{version}-%{release}

%if 0%{?rhel} == 7
Requires: python36-numpy
%else
Requires: python3-numpy
%endif

%description python
Quantitative estimate of elemental composition by spectroscopic and imaging techniques using X-ray fluorescence requires the availability of accurate data of X-ray interaction with matter. Although a wide number of computer codes and data sets are reported in literature, none of them is presented in the form of freely available library functions which can be easily included in software applications for X-ray fluorescence. This work presents a compilation of data sets from different published works and an xraylib interface in the form of callable functions. Although the target applications are on X-ray fluorescence, cross sections of interactions like photoionization, coherent scattering and Compton scattering, as well as form factors and anomalous scattering functions, are also available. 

This rpm package provides the python3 bindings of xraylib.

%package lua
Summary:A library for X-ray matter interactions cross sections for X-ray fluorescence applications: lua bindings
Requires: lua(abi) = %{lua_version}
Requires: %{name}%{?_isa} = %{version}-%{release}

%description lua
Quantitative estimate of elemental composition by spectroscopic and imaging techniques using X-ray fluorescence requires the availability of accurate data of X-ray interaction with matter. Although a wide number of computer codes and data sets are reported in literature, none of them is presented in the form of freely available library functions which can be easily included in software applications for X-ray fluorescence. This work presents a compilation of data sets from different published works and an xraylib interface in the form of callable functions. Although the target applications are on X-ray fluorescence, cross sections of interactions like photoionization, coherent scattering and Compton scattering, as well as form factors and anomalous scattering functions, are also available. 

This rpm package provides the lua bindings of xraylib.

%package ruby
Summary:A library for X-ray matter interactions cross sections for X-ray fluorescence applications: ruby bindings
Requires: ruby(release)
Requires: %{name}%{?_isa} = %{version}-%{release}

%description ruby
Quantitative estimate of elemental composition by spectroscopic and imaging techniques using X-ray fluorescence requires the availability of accurate data of X-ray interaction with matter. Although a wide number of computer codes and data sets are reported in literature, none of them is presented in the form of freely available library functions which can be easily included in software applications for X-ray fluorescence. This work presents a compilation of data sets from different published works and an xraylib interface in the form of callable functions. Although the target applications are on X-ray fluorescence, cross sections of interactions like photoionization, coherent scattering and Compton scattering, as well as form factors and anomalous scattering functions, are also available. 

This rpm package provides the ruby bindings of xraylib.

%package perl
Summary:A library for X-ray matter interactions cross sections for X-ray fluorescence applications: perl bindings
Requires: perl %{name}%{?_isa} = %{version}-%{release}

%description perl 
Quantitative estimate of elemental composition by spectroscopic and imaging techniques using X-ray fluorescence requires the availability of accurate data of X-ray interaction with matter. Although a wide number of computer codes and data sets are reported in literature, none of them is presented in the form of freely available library functions which can be easily included in software applications for X-ray fluorescence. This work presents a compilation of data sets from different published works and an xraylib interface in the form of callable functions. Although the target applications are on X-ray fluorescence, cross sections of interactions like photoionization, coherent scattering and Compton scattering, as well as form factors and anomalous scattering functions, are also available. 

This rpm package provides the perl bindings of xraylib.

%package php
Summary:A library for X-ray matter interactions cross sections for X-ray fluorescence applications: php bindings
Requires: php %{name}%{?_isa} = %{version}-%{release}

%description php
Quantitative estimate of elemental composition by spectroscopic and imaging techniques using X-ray fluorescence requires the availability of accurate data of X-ray interaction with matter. Although a wide number of computer codes and data sets are reported in literature, none of them is presented in the form of freely available library functions which can be easily included in software applications for X-ray fluorescence. This work presents a compilation of data sets from different published works and an xraylib interface in the form of callable functions. Although the target applications are on X-ray fluorescence, cross sections of interactions like photoionization, coherent scattering and Compton scattering, as well as form factors and anomalous scattering functions, are also available. 

This rpm package provides the PHP bindings of xraylib.


%prep

%setup -q

%build
%configure --disable-java --disable-idl --enable-fortran2003 --enable-python --enable-lua --enable-ruby --enable-ruby-integration --enable-perl-integration --enable-perl --enable-python-numpy --enable-php --enable-php-integration --enable-static FC=gfortran PYTHON=%{__python3} PERL=%{__perl} CYTHON=%{cython3}
#necessary to fix rpath issues during rpmbuild
sed -i 's|^hardcode_libdir_flag_spec=.*|hardcode_libdir_flag_spec=""|g' libtool
sed -i 's|^runpath_var=LD_RUN_PATH|runpath_var=DIE_RPATH_DIE|g' libtool

make 

%install
rm -rf $RPM_BUILD_ROOT
make install DESTDIR=$RPM_BUILD_ROOT

libtool --finish $RPM_BUILD_ROOT%{_libdir}
rm -f $RPM_BUILD_ROOT%{_libdir}/*.la
rm -f $RPM_BUILD_ROOT%{lua_libdir}/*.la
rm -f $RPM_BUILD_ROOT%{ruby_vendorarchdir}/*.la
rm -f $RPM_BUILD_ROOT%{perl_vendor_autolib}/xraylib/*.la
rm -f $RPM_BUILD_ROOT%{php_extdir}/*.la
rm -f $RPM_BUILD_ROOT%{python3_sitearch}/*.la

mkdir -p $RPM_BUILD_ROOT/%{_sysconfdir}/php.d
cat >$RPM_BUILD_ROOT/%{_sysconfdir}/php.d/xraylib.ini <<EOF
extension=xraylib.so
EOF

chmod 0644 $RPM_BUILD_ROOT/%{_sysconfdir}/php.d/xraylib.ini

%clean
rm -rf $RPM_BUILD_ROOT

%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig


%files
%defattr(-,root,root)

%{_libdir}/libxrl.so.*
%{_prefix}/share/xraylib/*.txt

%files devel
%defattr(-,root,root)
%{_libdir}/libxrl.so
%{_libdir}/libxrl.a
%{_includedir}/xraylib/*.h
%{_libdir}/pkgconfig/libxrl.pc

%files fortran
%defattr(-,root,root)

%{_libdir}/libxrlf03.so.*
%{_libdir}/libxrlf03.so
%{_libdir}/libxrlf03.a
%{_includedir}/xraylib/*.mod
%{_libdir}/pkgconfig/libxrlf03.pc

%files python
%defattr(-,root,root)
%{python3_sitelib}/xraylib.py*
%{python3_sitelib}/__pycache__/*
%{python3_sitearch}/_xraylib.*
%{python3_sitearch}/xraylib_np.*

%files lua
%defattr(-,root,root)
%{lua_libdir}/*

%files ruby
%defattr(-,root,root)
%{ruby_vendorarchdir}/*

%files perl
%defattr(-,root,root)
%{perl_vendor_archlib}/xraylib.pm
%{perl_vendor_autolib}/xraylib/xraylib.so

%files php
%defattr(-,root,root)
%{php_extdir}/xraylib.so
%{_datadir}/php/xraylib.php
%config(noreplace) %{_sysconfdir}/php.d/xraylib.ini

%changelog
* Sun May 16 2021 Tom Schoonjans
- Remove python2 support
- Do not use automake variables, switch to RPM macros

* Wed Sep 4 2019 Tom Schoonjans
- Remove RHEL6 support
- Fix python packaging bugs
- Extract python2 bindings into a separate package

* Mon Jul 23 2018 Tom Schoonjans
- Fix dependencies on core package

* Fri Jul 20 2018 Tom Schoonjans
- Remove python command line executable

* Wed May 2 2018 Tom Schoonjans
- Fix Fedora python build

* Mon Feb 12 2018 Tom Schoonjans
- Add PHP bindings

* Mon Sep 18 2017 Tom Schoonjans
- Remove python bindings for RHEL6

* Wed Nov 23 2016 Tom Schoonjans
- Remove support for IDL

* Wed Dec 10 2014 Tom Schoonjans
- Support added for python3

* Tue Jun 3 2014 Tom Schoonjans
- Added python-numpy bindings

* Mon Jun 24 2013 Tom Schoonjans
- Added perl bindings
- Slight change of python bindings

* Tue Jun 18 2013 Tom Schoonjans
- Added ruby bindings

* Thu Aug 23 2012 Tom Schoonjans
- Added lua bindings

* Wed May 18 2011 Tom Schoonjans
- Added xraylib_auger.pro to idl package

* Mon Mar 22 2010 Tom Schoonjans
- Python files modified so they don't contain java files

* Mon Mar 08 2010 Tom Schoonjans
- Added IDL bindings
- ldconfig line

* Thu Aug 20 2009 Tom Schoonjans
- Initial spec file. Perl bindings not included.
