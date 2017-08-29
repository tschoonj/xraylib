from __future__ import print_function
import numpy as np

def cshift(arr, offset):
	return np.roll(arr, offset)

def deriv(x, y):
	if len(x.shape) != 1 or x.shape[0] != y.shape[0]:
		raise TypeError("deriv: x and y must be 1D numpy arrays of the same length")

	x0 = cshift(x, 1)
	x2 = cshift(x, -1)
	x01 = x0 - x
	x02 = x0 - x2
	x12 = x - x2

	d = cshift(y, 1) * (x12 / (x01 * x02)) + \
		y * (1.0/x12 - 1.0/x01) - \
		cshift(y, -1) * (x01 / (x02 * x12))

	d[0] = y[0] * (x01[1]+x02[1])/(x01[1]*x02[1]) - \
		y[1] * x02[1]/(x01[1]*x12[1]) + \
		y[2] * x01[1]/(x02[1]*x12[1])

	d[-1] = -y[-3] * x12[-2]/(x01[-2]*x02[-2]) + \
		y[-2] * x02[-2]/(x01[-2]*x12[-2]) - \
		y[-1] * (x02[-2]+x12[-2]) / (x02[-2]*x12[-2])
	return d

if __name__ == "__main__":
	x = np.linspace(0.0, 10.0, 1001, dtype=np.float64)
	y = np.sin(x)
	dy = deriv(x, y)
	print("max diff {}".format(np.max(np.abs(dy - np.cos(x)))))
	
