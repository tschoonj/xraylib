PROGRAM xrltest

USE xraylib

IMPLICIT NONE

CALL XRayInit()
CALL SetHardExit(1)

WRITE (6,'(A)') 'Example of fortran program using xraylib'
WRITE (6,'(A,F12.6)') 'Ca K-alpha Fluorescence Line Energy: ',LineEnergy(20,KA_LINE);
WRITE (6,'(A,F12.6)') 'Fe partial photoionization cs of L3 at 6.0 keV: ',CS_Photo_Partial(26,L3_SHELL,6.0)
WRITE (6,'(A,F12.6)') 'Zr L1 edge energy: ',EdgeEnergy(40,L1_SHELL)
WRITE (6,'(A,F12.6)') 'Pb Lalpha XRF production cs at 20.0 keV (jump approx): ',CS_FluorLine(82,LA_LINE,20.0)
WRITE (6,'(A,F12.6)') 'Pb Lalpha XRF production cs at 20.0 keV (Kissel): ',CS_FluorLine_Kissel(82,LA_LINE,20.0)


ENDPROGRAM
