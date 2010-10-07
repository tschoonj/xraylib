from distutils.core import setup
import py2exe

setup(
	name="xraylib",
	version="2.14.0",
	author="Tom Schoonjans",
	console=['xraylib.py'],
	data_files=[('Doc',['C:\\Program Files (x86)\\xraylib\\Doc\\xraybanner.txt']),('Doc',['C:\\Program Files (x86)\\xraylib\\Doc\\xraydoc.txt']),('Doc',['C:\\Program Files (x86)\\xraylib\\Doc\\xrayfunc.txt']),('Doc',['C:\\Program Files (x86)\\xraylib\\Doc\\xrayhelp.txt']),("Microsoft.VC90.CRT",['msvcr90.dll','Microsoft.VC90.CRT.manifest'])   ],
	options={
		"py2exe":{
			"unbuffered": True,
			"optimize": 2,
			"bundle_files": 1,
			"includes":"_xraylib",
			"dll_excludes":['w9xpopen.exe']
	        }
        },
	zipfile=None
)
