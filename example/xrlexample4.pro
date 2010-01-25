
;Copyright (c) 2009, Tom Schoonjans
;All rights reserved.

;Redistribution and use in source and binary forms, with or without
;modification, are permitted provided that the following conditions are met:
;    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
;    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
;    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

;THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


;initialize the xraylib variables
;since this batch script will run in the $MAIN$, there is no need to call the xraylib common block
;however if you would want access to the xraylib variables from within a function or procedure, then you must call it

@xraylib

PRINT,'Example of IDL program using xraylib'
PRINT,'Ca K-alpha Fluorescence Line Energy: ',LineEnergy(20,KA_LINE)
PRINT,'Fe partial photoionization cs of L3 at 6.0 keV: ',CS_Photo_Partial(26,L3_SHELL,6.0)
PRINT,'Zr L1 edge energy: ',EdgeEnergy(40,L1_SHELL)
PRINT,'Pb Lalpha XRF production cs at 20.0 keV (jump approx): ',CS_FluorLine(82,LA_LINE,20.0)
PRINT,'Pb Lalpha XRF production cs at 20.0 keV (Kissel): ',CS_FluorLine_Kissel(82,LA_LINE,20.0)
PRINT,'Bi M1N2 radiative rate: ',RadRate(83,M1N2_LINE)

cdtest = CompoundParser('Ca(HCO3)2')
PRINT,'Ca(HCO3)2 contains ',cdtest.nAtomsAll, ' and ',cdtest.nElements,' elements'
FOR i=0L,cdtest.nElements-1 DO PRINT,'Element ',cdtest.Elements[i],' : ',cdtest.massFractions[i]*100.0,' %'

cdtest = CompoundParser('SiO2')
PRINT,'SiO2 contains ',cdtest.nAtomsAll, ' and ',cdtest.nElements,' elements'
FOR i=0L,cdtest.nElements-1 DO PRINT,'Element ',cdtest.Elements[i],' : ',cdtest.massFractions[i]*100.0,' %'

PRINT,'Ca(HCO3)2 Rayleigh cs at 10.0 keV: ',CS_Rayl_CP("Ca(HCO3)2",10.0) 


;the value of !ERROR_STATE will determine the exit status of IDL and therefore the outcome of make check
IF !ERROR_STATE.CODE eq 0 THEN EXIT,STATUS=0 ELSE EXIT,STATUS=1
