from _xraylib import *
import sys, string


if __name__ == '__main__' :
	XRayInit()
	print "Example of python program using xraylib"
	print "Ca K-alpha Fluorescence Line Energy: %f" % LineEnergy(20,KA_LINE)
	print "Fe partial photoionization cs of L3 at 6.0 keV: %f" % CS_Photo_Partial(26,L3_SHELL,6.0)
	print "Zr L1 edge energy: %f" % EdgeEnergy(40,L1_SHELL)
	print "Pb Lalpha XRF production cs at 20.0 keV (jump approx): %f" % CS_FluorLine(82,LA_LINE,20.0)
	print "Pb Lalpha XRF production cs at 20.0 keV (Kissel): %f" % CS_FluorLine_Kissel(82,LA_LINE,20.0)
	sys.exit(0)
