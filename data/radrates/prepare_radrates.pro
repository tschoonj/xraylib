;Copyright (c) 2009, Tom Schoonjans
;All rights reserved.

;Redistribution and use in source and binary forms, with or without
;modification, are permitted provided that the following conditions are met:
;    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
;    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
;    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

;THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



;little IDL script that produces a radrate.dat file based on a number of SPEC type files

radfiles = ['KShellRates.dat', 'LShellRates.dat','MShellRates.dat']

line=''
struct_template = {element:0, transition:'',probability:0.0D}
nstructs=0L

FOR i=0, N_ELEMENTS(radfiles)-1 DO BEGIN
	PRINT,'File: ',radfiles[i]
	;open file for reading
	OPENR,lun,radfiles[i],/GET_LUN
	WHILE (NOT EOF(lun)) DO BEGIN 
		READF,lun,line
		SWITCH STRMID(line,0,1) OF
			'#': BEGIN
				IF (STRMID(line,0,2) EQ '#L') THEN BEGIN
					tags=STRSPLIT(line,/EXTRACT)
					print,tags
				;	tags=tags[2:N_ELEMENTS(tags)-1]
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
				FOR j=1,N_ELEMENTS(values)-1 DO BEGIN
					IF (nstructs EQ 0) THEN BEGIN
						my_data = [REPLICATE(struct_template,1)]
					ENDIF ELSE BEGIN
						my_data = [my_data,REPLICATE(struct_template,1)]
					ENDELSE
					my_data[nstructs].element = FIX(values[0])
					my_data[nstructs].transition = tags[j+1]
					my_data[nstructs].probability= values[j]
					nstructs++
				ENDFOR
			END
		ENDSWITCH
	ENDWHILE
	FREE_LUN,lun
ENDFOR

;copy to new file
OPENW,lun,'radrate.dat',/GET_LUN
FOR i=0L,nstructs-1 DO BEGIN
	IF (my_data[i].probability GT 0.0D AND my_data[i].transition NE 'TOTAL') THEN PRINTF,lun,my_data[i].element,my_data[i].transition,my_data[i].probability,format='(I3,A6,G14.6)'
ENDFOR
FREE_LUN,lun



END
