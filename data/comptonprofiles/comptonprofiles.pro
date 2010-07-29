;this script will generate the xraylib file containing the compton profiles as well as the second derivatives for future interpolation

PRO comptonprofiles

data_template={element:0, N:0, UOCCUP: PTR_NEW(), UBIND: PTR_NEW(),pz: PTR_NEW(), total:ptr_new(), partial:ptr_new(),total_secderiv: PTR_NEW(),partial_secderiv: PTR_NEW() }
nstructs=0L
line=''
data_reader=0

OPENR,lun,'ComptonProfiles.dat',/GET_LUN

WHILE(NOT EOF(lun)) DO BEGIN
	READF,lun,line
	line=STRTRIM(line,1)
	SWITCH STRMID(line,0,1) OF
		'#': BEGIN
			CASE STRMID(line,0,3) OF
				'#S ': BEGIN
					IF (data_reader EQ 1) THEN BEGIN 
;						PRINT,'nstructs: ',nstructs
;						PRINT,'N_ELEMENTS(my_data)',N_ELEMENTS(my_data)
;						PRINT,'my_data[nstructs].N-2',my_data[nstructs].N-2
;						PRINT,'N_ELEMENTS(*(my_data[nstructs].partial))/(my_data[nstructs].N-2)',N_ELEMENTS(*(my_data[nstructs].partial))/(my_data[nstructs].N-2)
						*(my_data[nstructs].partial) = REFORM(*(my_data[nstructs].partial), my_data[nstructs].N-2,N_ELEMENTS(*(my_data[nstructs].partial))/(my_data[nstructs].N-2),/overwrite)
						;calculate second derivatives
						my_data[nstructs].total_secderiv = PTR_NEW(DERIV(*(my_data[nstructs].pz),DERIV(*(my_data[nstructs].pz),*(my_data[nstructs].total))))
						my_data[nstructs].partial_secderiv = PTR_NEW(DBLARR(my_data[nstructs].N-2,N_ELEMENTS(*(my_data[nstructs].partial))/(my_data[nstructs].N-2)))
						FOR i=0,my_data[nstructs].N-3 DO BEGIN
							(*(my_data[nstructs].partial_secderiv))[i,*] = DERIV(*(my_data[nstructs].pz),DERIV(*(my_data[nstructs].pz),(*(my_data[nstructs].partial))[i,*]))
						ENDFOR
						nstructs++
						data_reader=0

					ENDIF
					IF (nstructs EQ 0) THEN BEGIN
						my_data = [REPLICATE(data_template,1)]
					ENDIF ELSE BEGIN
						my_data = [my_data,REPLICATE(data_template,1)]
					ENDELSE
					splitted=STRSPLIT(line,/EXTRACT)
					my_data[nstructs].element=FIX(splitted[1])
				END
				'#N ':BEGIN
					splitted=STRSPLIT(line,/EXTRACT)
					my_data[nstructs].N=FIX(splitted[1])
				END
				'#UO':BEGIN
					splitted=STRSPLIT(line,/EXTRACT)
					my_data[nstructs].UOCCUP=PTR_NEW(FIX(splitted[1:my_data[nstructs].N-2]))
				END
				'#UB':BEGIN
					splitted=STRSPLIT(line,/EXTRACT)
					my_data[nstructs].UBIND=PTR_NEW(DOUBLE(splitted[1:my_data[nstructs].N-2]))
				END
				'#L ': BEGIN
					data_reader=1
				END
				ELSE:
			ENDCASE
			BREAK
		END
		'0':
		'1':
		'2':
		'3':
		'4':
		'5':
		'6':
		'7':
		'8':
		'9': BEGIN
			IF (NOT data_reader) THEN BEGIN
				PRINT,'data_reader error: aborting'
				PRINT,'Last string was: ',line
				return 
			ENDIF
			values=DOUBLE(STRSPLIT(line,/EXTRACT))
			IF (NOT PTR_VALID(my_data[nstructs].pz)) THEN BEGIN
				my_data[nstructs].pz = PTR_NEW([DOUBLE(values[0])])
				my_data[nstructs].total = PTR_NEW([DOUBLE(values[1])])
				my_data[nstructs].partial = PTR_NEW([DOUBLE(values[2:my_data[nstructs].N-1])])
			ENDIF ELSE BEGIN
				*(my_data[nstructs].pz) = [*(my_data[nstructs].pz),DOUBLE(values[0])]
				*(my_data[nstructs].total) = [*(my_data[nstructs].total),DOUBLE(values[1])]
				*(my_data[nstructs].partial) = [*(my_data[nstructs].partial),DOUBLE(values[2:my_data[nstructs].N-1])]
			ENDELSE
		END
	ENDSWITCH
ENDWHILE

*(my_data[nstructs].partial) = REFORM(*(my_data[nstructs].partial), my_data[nstructs].N-2,N_ELEMENTS(*(my_data[nstructs].partial))/(my_data[nstructs].N-2),/overwrite)
;calculate second derivatives
my_data[nstructs].total_secderiv = PTR_NEW(DERIV(*(my_data[nstructs].pz),DERIV(*(my_data[nstructs].pz),*(my_data[nstructs].total))))
my_data[nstructs].partial_secderiv = PTR_NEW(DBLARR(my_data[nstructs].N-2,N_ELEMENTS(*(my_data[nstructs].partial))/(my_data[nstructs].N-2)))
FOR i=0,my_data[nstructs].N-3 DO BEGIN
	(*(my_data[nstructs].partial_secderiv))[i,*] = DERIV(*(my_data[nstructs].pz),DERIV(*(my_data[nstructs].pz),(*(my_data[nstructs].partial))[i,*]))
ENDFOR
nstructs++

FREE_LUN,lun


;reading in seems to work
;so does calculating the second derivatives
;write them to a file

OPENW,lun,'../comptonprofiles.dat',/GET_LUN

FOR i=0,nstructs-1 DO BEGIN
	PRINTF,lun,my_data[i].N-2,N_ELEMENTS(*(my_data[i].pz))
	PRINTF,lun,*(my_data[i].UOCCUP)
	PRINTF,lun,*(my_data[i].UBIND)
	PRINTF,lun,*(my_data[i].PZ)
	PRINTF,lun,*(my_data[i].TOTAL)
	PRINTF,lun,*(my_data[i].TOTAL_SECDERIV)
	FOR j=0,my_data[i].N-3 DO BEGIN
		FOR k=0,N_ELEMENTS(*(my_data[i].pz))-1 DO BEGIN
			PRINTF,lun,(*(my_data[i].PARTIAL))[j,k]
		ENDFOR
	ENDFOR
	FOR j=0,my_data[i].N-3 DO BEGIN
		FOR k=0,N_ELEMENTS(*(my_data[i].pz))-1 DO BEGIN
			PRINTF,lun,(*(my_data[i].PARTIAL_SECDERIV))[j,k]
		ENDFOR
	ENDFOR
ENDFOR


FREE_LUN,lun



END
