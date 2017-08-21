#!/usr/bin/env/python

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils
import numpy as np
import xraylib as xrl

from storable import retrieve # to read Perl storables

def check_energy(energy):
	for i in range(1, energy.size):
		if (energy[i] == energy[i-1]):
			energy[i] += 1E-6
	return energy

data = retrieve('henke.db')

with open('../fi.dat', 'w') as fi, open('../fii.dat', 'w') as fii:
	for Z in range(1, 93):
		print('Processing {Z}'.format(Z=Z))
		symbol = xrl.AtomicNumberToSymbol(Z).lower()
		energy = check_energy(np.array(data[symbol]['energy'], dtype=np.float64))
		f1 = np.array(data[symbol]['f1'], dtype=np.float64)
		f1dd = utils.deriv(energy, utils.deriv(energy, f1))
		f2 = np.array(data[symbol]['f2'], dtype=np.float64)
		f2dd = utils.deriv(energy, utils.deriv(energy, f2))
		npoints = energy.size
		fi.write('{n}\n'.format(n=npoints))
		fii.write('{n}\n'.format(n=npoints))
		for point in range(npoints):
			fi.write('{0}\t{1}\t{2}\n'.format(energy[point], f1[point], f1dd[point]))
			fii.write('{0}\t{1}\t{2}\n'.format(energy[point], f2[point], f2dd[point]))
