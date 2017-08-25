#!/usr/bin/env python

import requests
import numpy as np
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils

def chunkstring(string, length):
	return (string[0+i:length+i] for i in range(0, len(string), length))

def get_double(string):
	string = string.strip()
	if string == "0.0":
		return 0.0
	plus_pos = string.rfind('+')
	min_pos = string.rfind('-')
	if plus_pos > -1:
		new_string = string[0:plus_pos] + 'E' + string[plus_pos+1:]
	elif min_pos > 0:
		new_string = string[0:min_pos] + 'E' + string[min_pos:]
	else:
		new_string = string
	return float(new_string)

with open('../fi.dat', 'w') as fiF, open('../fii.dat', 'w') as fiiF:
	for Z in range(1, 101):
		url = "https://www-nds.iaea.org/epdl97/data/anomlous/za{:03d}000".format(Z)
		print(url)
		r = requests.get(url)
		r.raise_for_status()
		contents = r.text.splitlines()
		energies = []
		fis = []
		fiis = []
		for line in contents[2:]:
			# lines should be 9 * 10 characters
			chunks = list(chunkstring(line, 10))
			#print(chunks)
			# needed are cols 0 (Energy), 1 (Fi), 5 (Fii)
			energy = get_double(chunks[0])
			fi = get_double(chunks[1])
			fii = -1.0 * get_double(chunks[5])
			#print("energy {}, fi {}, fii {}".format(energy, fi, fii))
			energies.append(energy)
			fis.append(fi)
			fiis.append(fii)

		energies = np.array(energies, dtype=np.float64)*1000.0
		fis = np.array(fis, dtype=np.float64)
		fiis = np.array(fiis, dtype=np.float64)
		
		fis2 = utils.deriv(energies, utils.deriv(energies, fis))
		fis2[np.where(np.logical_or(fis2 < -1, fis2 > 1))] = 0.0
		fiis2 = utils.deriv(energies, utils.deriv(energies, fiis))
		fiis2[np.where(np.logical_or(fiis2 < -1, fiis2 > 1))] = 0.0

		npoints = energies.size
		fiF.write('{n}\n'.format(n=npoints))
		fiiF.write('{n}\n'.format(n=npoints))
		for point in range(npoints):
			fiF.write('{0}\t{1}\t{2}\n'.format(energies[point], fis[point], fis2[point]))
			fiiF.write('{0}\t{1}\t{2}\n'.format(energies[point], fiis[point], fiis2[point]))
