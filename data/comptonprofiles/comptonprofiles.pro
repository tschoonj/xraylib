;this script will generate the xraylib file containing the compton profiles as well as the second derivatives for future interpolation

PRO comptonprofiles

COMMON xraylib

full_elec_conf = [2,$
	2,2,4,$
	2,2,4,4,6,$
	2,2,4,4,6,6,8,$
	2,2,4,4,6,6,8,$
	2,2,4,4,6,$
	2,2,4,$
	2$
	]


;read in correct biggs electronic configurations
biggs_elec_config = []

OPENR,lun,'biggs_electron_configuration.dat',/get_lun
line=''
WHILE(NOT EOF(lun)) DO BEGIN
	READF,lun,line
	line=STRTRIM(line,1)
	IF (strmid(line,0,1) NE '#') THEN BEGIN
		biggs_elec_config = [biggs_elec_config,PTR_NEW(DOUBLE(STRSPLIT(line,/EXTRACT)))]
	ENDIF
ENDWHILE
FREE_LUN,lun

FOR i=0,N_ELEMENTS(biggs_elec_config)-1 DO BEGIN
	IF (TOTAL(*(biggs_elec_config[i])) NE i+1) THEN BEGIN
		PRINT,'Error in biggs_electron_configuration.dat for element: ',i+1
		STOP
	ENDIF
ENDFOR







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
						*(my_data[nstructs].partial) = ALOG(REFORM(*(my_data[nstructs].partial), my_data[nstructs].N-2,N_ELEMENTS(*(my_data[nstructs].partial))/(my_data[nstructs].N-2),/overwrite))
						*(my_data[nstructs].total) = ALOG(*(my_data[nstructs].total))
						*(my_data[nstructs].pz)=ALOG(*(my_data[nstructs].pz)+1)
						;calculate second derivatives
						my_data[nstructs].total_secderiv = PTR_NEW(DERIV(*(my_data[nstructs].pz),DERIV(*(my_data[nstructs].pz),*(my_data[nstructs].total))))
						my_data[nstructs].partial_secderiv = PTR_NEW(DBLARR(my_data[nstructs].N-2,N_ELEMENTS(*(my_data[nstructs].partial))/(my_data[nstructs].N-2)))
						FOR i=0,my_data[nstructs].N-3 DO BEGIN
							(*(my_data[nstructs].partial_secderiv))[i,*] = DERIV(*(my_data[nstructs].pz),DERIV(*(my_data[nstructs].pz),(*(my_data[nstructs].partial))[i,*]))
						ENDFOR
						;adapt partial comptonprofiles
						print,'n elec_config:',N_ELEMENTS(*(biggs_elec_config[nstructs]))
						print,'elec config:',*(biggs_elec_config[nstructs])
						print,'n uoccup:',N_ELEMENTS(*(my_data[nstructs].UOCCUP))
						print,'uoccup:',*(my_data[nstructs].UOCCUP)
						IF (N_ELEMENTS(*(biggs_elec_config[nstructs])) NE N_ELEMENTS(*(my_data[nstructs].UOCCUP))) THEN BEGIN
							j=0
							partial_new=DBLARR(N_ELEMENTS(*(biggs_elec_config[nstructs])),N_ELEMENTS(*(my_data[nstructs].pz)))
							partial_secderiv_new=DBLARR(N_ELEMENTS(*(biggs_elec_config[nstructs])),N_ELEMENTS(*(my_data[nstructs].pz)))
							for i=0,N_ELEMENTS(*(my_data[nstructs].UOCCUP))-1 DO BEGIN
								print,'i:',i
								print,'j:',j
								print,'elec config j:',(*(biggs_elec_config[nstructs]))[j]
								print,'uoccup',(*(my_data[nstructs].UOCCUP))[i]
								my_zeroes = DBLARR(1,N_ELEMENTS(*(my_data[nstructs].pz)))
								WHILE ((*(biggs_elec_config[nstructs]))[j] EQ 0.0) DO BEGIN
									partial_new[j,*] = my_zeroes
									j = j+1
								ENDWHILE
								IF ((*(biggs_elec_config[nstructs]))[j] LT (*(my_data[nstructs].UOCCUP))[i]) THEN BEGIN
									partial_new[j,*] = (*(my_data[nstructs].partial))[i,*]	
									partial_new[j+1,*] = (*(my_data[nstructs].partial))[i,*]	
									partial_secderiv_new[j,*] = (*(my_data[nstructs].partial_secderiv))[i,*]	
									partial_secderiv_new[j+1,*] = (*(my_data[nstructs].partial_secderiv))[i,*]	
									j = j+2
								ENDIF ELSE BEGIN
									partial_new[j,*] = (*(my_data[nstructs].partial))[i,*]	
									partial_secderiv_new[j,*] = (*(my_data[nstructs].partial_secderiv))[i,*]	
									j = j+1
									
								ENDELSE

							ENDFOR
							ptr_free,my_data[nstructs].partial
							ptr_free,my_data[nstructs].partial_secderiv
							ptr_free,my_data[nstructs].UOCCUP
							my_data[nstructs].partial = PTR_NEW(TEMPORARY(partial_new))
							my_data[nstructs].partial_secderiv = PTR_NEW(TEMPORARY(partial_secderiv_new))
							my_data[nstructs].N=N_ELEMENTS(*(biggs_elec_config[nstructs]))+2
							my_data[nstructs].uoccup = biggs_elec_config[nstructs] 


						ENDIF


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
					print,'Element: ',my_data[nstructs].element
				END
				'#N ':BEGIN
					splitted=STRSPLIT(line,/EXTRACT)
					my_data[nstructs].N=FIX(splitted[1])
				END
				'#UO':BEGIN
					splitted=STRSPLIT(line,/EXTRACT)
					my_data[nstructs].UOCCUP=PTR_NEW(FIX(splitted[1:my_data[nstructs].N-2]))
					;analyze electronic configuration
					;construct kissel array
					;elec_config=[]
					;for i=K_SHELL,Q3_SHELL do begin
					;	temp_value= ElectronConfig(my_data[nstructs].element,i)
					;	elec_config = [elec_config, temp_value]
					;endfor
					;nonzero = WHERE(elec_config GT 0.0)
					;elec_config = elec_config[0:nonzero[-1]]
					;IF (N_ELEMENTS(elec_config) EQ N_ELEMENTS(*(my_data[nstructs].UOCCUP))) THEN BEGIN
					;	PRINT,'Electronic configuration OK for '+STRING(my_data[nstructs].element)
					;ENDIF

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

*(my_data[nstructs].partial) = ALOG(REFORM(*(my_data[nstructs].partial), my_data[nstructs].N-2,N_ELEMENTS(*(my_data[nstructs].partial))/(my_data[nstructs].N-2),/overwrite))
*(my_data[nstructs].total) = ALOG(*(my_data[nstructs].total))
*(my_data[nstructs].pz)=ALOG(*(my_data[nstructs].pz)+1)
;calculate second derivatives
my_data[nstructs].total_secderiv = PTR_NEW(DERIV(*(my_data[nstructs].pz),DERIV(*(my_data[nstructs].pz),*(my_data[nstructs].total))))
my_data[nstructs].partial_secderiv = PTR_NEW(DBLARR(my_data[nstructs].N-2,N_ELEMENTS(*(my_data[nstructs].partial))/(my_data[nstructs].N-2)))
FOR i=0,my_data[nstructs].N-3 DO BEGIN
	(*(my_data[nstructs].partial_secderiv))[i,*] = DERIV(*(my_data[nstructs].pz),DERIV(*(my_data[nstructs].pz),(*(my_data[nstructs].partial))[i,*]))
ENDFOR
IF (N_ELEMENTS(*(biggs_elec_config[nstructs])) NE N_ELEMENTS(*(my_data[nstructs].UOCCUP))) THEN BEGIN
	j=0
	partial_new=DBLARR(N_ELEMENTS(*(biggs_elec_config[nstructs])),N_ELEMENTS(*(my_data[nstructs].pz)))
	partial_secderiv_new=DBLARR(N_ELEMENTS(*(biggs_elec_config[nstructs])),N_ELEMENTS(*(my_data[nstructs].pz)))
	for i=0,N_ELEMENTS(*(my_data[nstructs].UOCCUP))-1 DO BEGIN
		my_zeroes = DBLARR(1,N_ELEMENTS(*(my_data[nstructs].pz)))
		WHILE ((*(biggs_elec_config[nstructs]))[j] EQ 0.0) DO BEGIN
			partial_new[j,*] = my_zeroes
			j = j+1
		ENDWHILE
		IF ((*(biggs_elec_config[nstructs]))[j] LT (*(my_data[nstructs].UOCCUP))[i]) THEN BEGIN
			partial_new[j,*] = (*(my_data[nstructs].partial))[i,*]	
			partial_new[j+1,*] = (*(my_data[nstructs].partial))[i,*]	
			partial_secderiv_new[j,*] = (*(my_data[nstructs].partial_secderiv))[i,*]	
			partial_secderiv_new[j+1,*] = (*(my_data[nstructs].partial_secderiv))[i,*]	
			j = j+2
		ENDIF ELSE BEGIN
			partial_new[j,*] = (*(my_data[nstructs].partial))[i,*]	
			partial_secderiv_new[j,*] = (*(my_data[nstructs].partial_secderiv))[i,*]	
			j = j+1
								
		ENDELSE

	ENDFOR
	ptr_free,my_data[nstructs].partial
	ptr_free,my_data[nstructs].partial_secderiv
	my_data[nstructs].partial = PTR_NEW(TEMPORARY(partial_new))
	my_data[nstructs].partial_secderiv = PTR_NEW(TEMPORARY(partial_secderiv_new))
	my_data[nstructs].uoccup = biggs_elec_config[nstructs] 
	my_data[nstructs].N=N_ELEMENTS(*(biggs_elec_config[nstructs]))+2



ENDIF
nstructs++

FREE_LUN,lun


;reading in seems to work
;so does calculating the second derivatives
;write them to a file

OPENW,lun,'../comptonprofiles.dat',/GET_LUN

FOR i=0,nstructs-1 DO BEGIN
	PRINTF,lun,my_data[i].N-2,N_ELEMENTS(*(my_data[i].pz))
	PRINTF,lun,FIX(*(my_data[i].UOCCUP))
	PRINTF,lun,*(my_data[i].PZ)
	PRINTF,lun,*(my_data[i].TOTAL)
	PRINTF,lun,*(my_data[i].TOTAL_SECDERIV)
	FOR j=0,my_data[i].N-3 DO BEGIN
		IF ((*(my_data[i].UOCCUP))[j] LE 0.0) THEN CONTINUE
		;FOR k=0,N_ELEMENTS(*(my_data[i].pz))-1 DO BEGIN
			PRINTF,lun,(*(my_data[i].PARTIAL))[j,*]
		;ENDFOR
	ENDFOR
	FOR j=0,my_data[i].N-3 DO BEGIN
		IF ((*(my_data[i].UOCCUP))[j] LE 0.0) THEN CONTINUE
		;FOR k=0,N_ELEMENTS(*(my_data[i].pz))-1 DO BEGIN
			PRINTF,lun,(*(my_data[i].PARTIAL_SECDERIV))[j,*]
		;ENDFOR
	ENDFOR
ENDFOR


FREE_LUN,lun



END
