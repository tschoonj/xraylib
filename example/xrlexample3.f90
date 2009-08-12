PROGRAM xrltest

USE xraylib

INTEGER (KIND=C_INT) :: Z = 20 

CALL XRayInit()

WRITE (6,'(A)') 'Example of fortran program using xraylib'
WRITE (6,'(A,F12.6)') 'Calcium K-alpha Fluorescence Line Energy: ',LineEnergy(Z,KA_LINE);

ENDPROGRAM
