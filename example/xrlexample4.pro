;initialize the xraylib variables
;since this batch script will run in the $MAIN$, there is no need to call the xraylib common block
;however if you would want access to the xraylib variables from within a function or procedure, then you must call it

@xraylib

PRINT,'Example of IDL program using xraylib'
PRINT,'Ca K-alpha Fluorescence Line Energy: ',LineEnergy(20,KA_LINE)
PRINT,'Fe partial photoionization cs of L3 at 6.0 keV: ',CS_Photo_Partial(26,L3_SHELL,6.0)
PRINT,'Zr L1 edge energy: ',EdgeEnergy(40,L1_SHELL)
PRINT,'Pb Lalpha XRF production cs at 20.0 keV (jump approx): ',CS_FluorLine(82,LA_LINE,20.0)
PRINT,'Pb Lalpha XRF production cs at 20.0 keV (Kissel): ',CS_FluorLine_Kissel(82,LA_LINE,20.0);

;the value of !ERROR_STATE will determine the exit status and therefore the outcome of make check
IF !ERROR_STATE.CODE eq 0 THEN EXIT,STATUS=0 ELSE EXIT,STATUS=1
