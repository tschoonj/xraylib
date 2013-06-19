;process cs_energy_perl.txt
;calculate second derivatives
;for future cubic spline interpolation


OPENR,lun,'cs_energy_perl.txt',/get_lun
nElements = 0

READF,lun,nElements

all_data = []

FOR i=0,nElements-1 DO BEGIN
	nEnergies = 0
	READF,lun,nEnergies
	data = {nEnergies:nEnergies, energies:DBLARR(nEnergies), cs:DBLARR(nEnergies), cs2:DBLARR(nEnergies)}
	energy=0.0D
	cs=0.0D
	FOR j=0,nEnergies-1 DO BEGIN
		READF,lun,energy,cs
		data.energies[j]=energy
		data.cs[j]=cs
	ENDFOR
	data.energies = ALOG(data.energies)
	data.cs = ALOG(data.cs)
	data.cs2 = DERIV(data.energies, DERIV(data.energies, data.cs))
	data.cs2[WHERE(data.cs2 LT -1.0D OR data.cs2 GT 1.0D)] = 0.0D
	all_data = [all_data, PTR_NEW(data)]
ENDFOR

FREE_LUN,lun

OPENW,lun,'../CS_Energy.dat', /GET_LUN
PRINTF, lun, nElements
FOR i=0, nElements-1 DO BEGIN
	PRINTF, lun, (*all_data[i]).nEnergies
	FOR j=0,(*all_data[i]).nEnergies-1 DO BEGIN
		PRINTF, lun, (*all_data[i]).energies[j],(*all_data[i]).cs[j], (*all_data[i]).cs2[j]
	ENDFOR
ENDFOR



END
