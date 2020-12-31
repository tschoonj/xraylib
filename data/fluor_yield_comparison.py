import numpy as np
import xraylib_np as xrl_np
import matplotlib.pyplot as plt
from matplotlib import gridspec

Z = np.arange(3, 99)

yields_bad = xrl_np.FluorYield(Z, np.array([xrl_np.K_SHELL])).squeeze()

yields_good = []

with open('fluor_yield_revised.dat', 'r') as file:
    for line in file:
        [myZ, y] = line.split()
        yields_good.append(y)
    yields_good = np.asarray(yields_good, dtype=np.float64)


fig = plt.figure(figsize=(7, 10))
gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
ax = fig.add_subplot(gs[0])
ax.set_xlabel('Atomic number')
ax.set_ylabel('Fluorescence yield')
plot_bad, = ax.plot(Z, yields_bad, color='r', marker='o', label='Fitted Hubbell')
plot_good, = ax.plot(Z, yields_good[0:Z.shape[0]], color='g', marker='o', label='Recommended Hubbell')
ax.legend(handles=[plot_bad, plot_good], loc=4)

ax = fig.add_subplot(gs[1])
ax.set_ylabel('Absolute diff')
ax.plot(Z, (yields_good[0:Z.shape[0]] - yields_bad), color='b', marker='o')

ax = fig.add_subplot(gs[2])
ax.set_ylabel('Relative diff')
ax.plot(Z, (yields_good[0:Z.shape[0]] - yields_bad) / yields_good[0:Z.shape[0]], color='b', marker='o')

plt.draw()
plt.savefig('yield-comparison.png')
plt.close()
