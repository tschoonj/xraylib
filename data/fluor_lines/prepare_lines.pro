;Copyright (c) 2009, Tom Schoonjans
;All rights reserved.

;Redistribution and use in source and binary forms, with or without
;modification, are permitted provided that the following conditions are met:
;    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
;    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
;    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

;THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



;little IDL script that produces a edges.dat file based on BindingEnergies.dat

inputfile='BindingEnergies.dat'
outputfile='fluor_lines.dat'
line=''

OPENR,lun,inputfile,/GET_LUN
OPENW,lun2,outputfile,/GET_LUN
WHILE (NOT EOF(lun)) DO BEGIN
	READF,lun,line
	SWITCH STRMID(line,0,1) OF
		'#': BEGIN
			IF (STRMID(line,0,2) EQ '#L') THEN BEGIN
				tags=STRSPLIT(line,/EXTRACT)
				print,tags
			ENDIF
			BREAK
		END
		'1':
		'2':
		'3':
		'4':
		'5':
		'6':
		'7':
		'8':
		'9': BEGIN
			values=DOUBLE(STRSPLIT(line,/EXTRACT))
			FOR i=0,N_ELEMENTS(values)-2 DO BEGIN
				FOR j=i+1,N_ELEMENTS(values)-2 DO BEGIN
					IF ((values[i+1]-values[j+1]) GT 0.0D AND values[j+1] GT 0.0D) THEN BEGIN
						PRINTF,lun2,FIX(values[0]),STRCOMPRESS(tags[i+2]+tags[j+2],/REMOVE_ALL),(values[i+1]-values[j+1])*1000.0,format='(I3,A6,G14.6)'	
					ENDIF
				ENDFOR
			ENDFOR
		END
	ENDSWITCH
ENDWHILE

FREE_LUN,lun
FREE_LUN,lun2

END

