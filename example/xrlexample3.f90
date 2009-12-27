
!Copyright (c) 2009, Tom Schoonjans
!All rights reserved.

!Redistribution and use in source and binary forms, with or without
!modification, are permitted provided that the following conditions are met:
!    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
!    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
!    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

!THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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
WRITE (6,'(A,F12.6)') 'Bi M1N2 radiative rate: ',RadRate(83,M1N2_LINE)


ENDPROGRAM
