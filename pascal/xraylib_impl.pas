{ This file has been generated automatically using generate-code.py}

function AtomicWeight_C(Z:longint; error:PPxrl_error):double;cdecl;external External_library name 'AtomicWeight';

function AtomicWeight(Z:longint):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := AtomicWeight_C(Z, @error);
    process_error(error);
    result := rv
end;

function ElementDensity_C(Z:longint; error:PPxrl_error):double;cdecl;external External_library name 'ElementDensity';

function ElementDensity(Z:longint):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := ElementDensity_C(Z, @error);
    process_error(error);
    result := rv
end;

function CS_Total_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_Total';

function CS_Total(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_Total_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function CS_Photo_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_Photo';

function CS_Photo(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_Photo_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function CS_Rayl_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_Rayl';

function CS_Rayl(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_Rayl_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function CS_Compt_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_Compt';

function CS_Compt(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_Compt_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function CS_Energy_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_Energy';

function CS_Energy(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_Energy_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function CS_KN_C(E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_KN';

function CS_KN(E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_KN_C(E, @error);
    process_error(error);
    result := rv
end;

function CSb_Total_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_Total';

function CSb_Total(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_Total_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function CSb_Photo_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_Photo';

function CSb_Photo(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_Photo_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function CSb_Rayl_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_Rayl';

function CSb_Rayl(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_Rayl_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function CSb_Compt_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_Compt';

function CSb_Compt(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_Compt_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function DCS_Thoms_C(theta:double; error:PPxrl_error):double;cdecl;external External_library name 'DCS_Thoms';

function DCS_Thoms(theta:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := DCS_Thoms_C(theta, @error);
    process_error(error);
    result := rv
end;

function DCS_KN_C(E:double; theta:double; error:PPxrl_error):double;cdecl;external External_library name 'DCS_KN';

function DCS_KN(E:double; theta:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := DCS_KN_C(E, theta, @error);
    process_error(error);
    result := rv
end;

function DCS_Rayl_C(Z:longint; E:double; theta:double; error:PPxrl_error):double;cdecl;external External_library name 'DCS_Rayl';

function DCS_Rayl(Z:longint; E:double; theta:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := DCS_Rayl_C(Z, E, theta, @error);
    process_error(error);
    result := rv
end;

function DCS_Compt_C(Z:longint; E:double; theta:double; error:PPxrl_error):double;cdecl;external External_library name 'DCS_Compt';

function DCS_Compt(Z:longint; E:double; theta:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := DCS_Compt_C(Z, E, theta, @error);
    process_error(error);
    result := rv
end;

function DCSb_Rayl_C(Z:longint; E:double; theta:double; error:PPxrl_error):double;cdecl;external External_library name 'DCSb_Rayl';

function DCSb_Rayl(Z:longint; E:double; theta:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := DCSb_Rayl_C(Z, E, theta, @error);
    process_error(error);
    result := rv
end;

function DCSb_Compt_C(Z:longint; E:double; theta:double; error:PPxrl_error):double;cdecl;external External_library name 'DCSb_Compt';

function DCSb_Compt(Z:longint; E:double; theta:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := DCSb_Compt_C(Z, E, theta, @error);
    process_error(error);
    result := rv
end;

function DCSP_Thoms_C(theta:double; phi:double; error:PPxrl_error):double;cdecl;external External_library name 'DCSP_Thoms';

function DCSP_Thoms(theta:double; phi:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := DCSP_Thoms_C(theta, phi, @error);
    process_error(error);
    result := rv
end;

function DCSP_KN_C(E:double; theta:double; phi:double; error:PPxrl_error):double;cdecl;external External_library name 'DCSP_KN';

function DCSP_KN(E:double; theta:double; phi:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := DCSP_KN_C(E, theta, phi, @error);
    process_error(error);
    result := rv
end;

function DCSP_Rayl_C(Z:longint; E:double; theta:double; phi:double; error:PPxrl_error):double;cdecl;external External_library name 'DCSP_Rayl';

function DCSP_Rayl(Z:longint; E:double; theta:double; phi:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := DCSP_Rayl_C(Z, E, theta, phi, @error);
    process_error(error);
    result := rv
end;

function DCSP_Compt_C(Z:longint; E:double; theta:double; phi:double; error:PPxrl_error):double;cdecl;external External_library name 'DCSP_Compt';

function DCSP_Compt(Z:longint; E:double; theta:double; phi:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := DCSP_Compt_C(Z, E, theta, phi, @error);
    process_error(error);
    result := rv
end;

function DCSPb_Rayl_C(Z:longint; E:double; theta:double; phi:double; error:PPxrl_error):double;cdecl;external External_library name 'DCSPb_Rayl';

function DCSPb_Rayl(Z:longint; E:double; theta:double; phi:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := DCSPb_Rayl_C(Z, E, theta, phi, @error);
    process_error(error);
    result := rv
end;

function DCSPb_Compt_C(Z:longint; E:double; theta:double; phi:double; error:PPxrl_error):double;cdecl;external External_library name 'DCSPb_Compt';

function DCSPb_Compt(Z:longint; E:double; theta:double; phi:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := DCSPb_Compt_C(Z, E, theta, phi, @error);
    process_error(error);
    result := rv
end;

function FF_Rayl_C(Z:longint; q:double; error:PPxrl_error):double;cdecl;external External_library name 'FF_Rayl';

function FF_Rayl(Z:longint; q:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := FF_Rayl_C(Z, q, @error);
    process_error(error);
    result := rv
end;

function SF_Compt_C(Z:longint; q:double; error:PPxrl_error):double;cdecl;external External_library name 'SF_Compt';

function SF_Compt(Z:longint; q:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := SF_Compt_C(Z, q, @error);
    process_error(error);
    result := rv
end;

function MomentTransf_C(E:double; theta:double; error:PPxrl_error):double;cdecl;external External_library name 'MomentTransf';

function MomentTransf(E:double; theta:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := MomentTransf_C(E, theta, @error);
    process_error(error);
    result := rv
end;

function LineEnergy_C(Z:longint; line:longint; error:PPxrl_error):double;cdecl;external External_library name 'LineEnergy';

function LineEnergy(Z:longint; line:longint):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := LineEnergy_C(Z, line, @error);
    process_error(error);
    result := rv
end;

function FluorYield_C(Z:longint; shell:longint; error:PPxrl_error):double;cdecl;external External_library name 'FluorYield';

function FluorYield(Z:longint; shell:longint):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := FluorYield_C(Z, shell, @error);
    process_error(error);
    result := rv
end;

function CosKronTransProb_C(Z:longint; trans:longint; error:PPxrl_error):double;cdecl;external External_library name 'CosKronTransProb';

function CosKronTransProb(Z:longint; trans:longint):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CosKronTransProb_C(Z, trans, @error);
    process_error(error);
    result := rv
end;

function EdgeEnergy_C(Z:longint; shell:longint; error:PPxrl_error):double;cdecl;external External_library name 'EdgeEnergy';

function EdgeEnergy(Z:longint; shell:longint):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := EdgeEnergy_C(Z, shell, @error);
    process_error(error);
    result := rv
end;

function JumpFactor_C(Z:longint; shell:longint; error:PPxrl_error):double;cdecl;external External_library name 'JumpFactor';

function JumpFactor(Z:longint; shell:longint):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := JumpFactor_C(Z, shell, @error);
    process_error(error);
    result := rv
end;

function CS_FluorLine_C(Z:longint; line:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_FluorLine';

function CS_FluorLine(Z:longint; line:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_FluorLine_C(Z, line, E, @error);
    process_error(error);
    result := rv
end;

function CSb_FluorLine_C(Z:longint; line:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_FluorLine';

function CSb_FluorLine(Z:longint; line:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_FluorLine_C(Z, line, E, @error);
    process_error(error);
    result := rv
end;

function CS_FluorShell_C(Z:longint; shell:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_FluorShell';

function CS_FluorShell(Z:longint; shell:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_FluorShell_C(Z, shell, E, @error);
    process_error(error);
    result := rv
end;

function CSb_FluorShell_C(Z:longint; shell:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_FluorShell';

function CSb_FluorShell(Z:longint; shell:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_FluorShell_C(Z, shell, E, @error);
    process_error(error);
    result := rv
end;

function RadRate_C(Z:longint; line:longint; error:PPxrl_error):double;cdecl;external External_library name 'RadRate';

function RadRate(Z:longint; line:longint):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := RadRate_C(Z, line, @error);
    process_error(error);
    result := rv
end;

function ComptonEnergy_C(E0:double; theta:double; error:PPxrl_error):double;cdecl;external External_library name 'ComptonEnergy';

function ComptonEnergy(E0:double; theta:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := ComptonEnergy_C(E0, theta, @error);
    process_error(error);
    result := rv
end;

function Fi_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'Fi';

function Fi(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := Fi_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function Fii_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'Fii';

function Fii(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := Fii_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function CS_Photo_Total_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_Photo_Total';

function CS_Photo_Total(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_Photo_Total_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function CSb_Photo_Total_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_Photo_Total';

function CSb_Photo_Total(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_Photo_Total_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function CS_Photo_Partial_C(Z:longint; shell:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_Photo_Partial';

function CS_Photo_Partial(Z:longint; shell:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_Photo_Partial_C(Z, shell, E, @error);
    process_error(error);
    result := rv
end;

function CSb_Photo_Partial_C(Z:longint; shell:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_Photo_Partial';

function CSb_Photo_Partial(Z:longint; shell:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_Photo_Partial_C(Z, shell, E, @error);
    process_error(error);
    result := rv
end;

function CS_Total_Kissel_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_Total_Kissel';

function CS_Total_Kissel(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_Total_Kissel_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function CSb_Total_Kissel_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_Total_Kissel';

function CSb_Total_Kissel(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_Total_Kissel_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function ComptonProfile_C(Z:longint; pz:double; error:PPxrl_error):double;cdecl;external External_library name 'ComptonProfile';

function ComptonProfile(Z:longint; pz:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := ComptonProfile_C(Z, pz, @error);
    process_error(error);
    result := rv
end;

function ComptonProfile_Partial_C(Z:longint; shell:longint; pz:double; error:PPxrl_error):double;cdecl;external External_library name 'ComptonProfile_Partial';

function ComptonProfile_Partial(Z:longint; shell:longint; pz:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := ComptonProfile_Partial_C(Z, shell, pz, @error);
    process_error(error);
    result := rv
end;

function ElectronConfig_C(Z:longint; shell:longint; error:PPxrl_error):double;cdecl;external External_library name 'ElectronConfig';

function ElectronConfig(Z:longint; shell:longint):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := ElectronConfig_C(Z, shell, @error);
    process_error(error);
    result := rv
end;

function ElectronConfig_Biggs_C(Z:longint; shell:longint; error:PPxrl_error):double;cdecl;external External_library name 'ElectronConfig_Biggs';

function ElectronConfig_Biggs(Z:longint; shell:longint):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := ElectronConfig_Biggs_C(Z, shell, @error);
    process_error(error);
    result := rv
end;

function AtomicLevelWidth_C(Z:longint; shell:longint; error:PPxrl_error):double;cdecl;external External_library name 'AtomicLevelWidth';

function AtomicLevelWidth(Z:longint; shell:longint):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := AtomicLevelWidth_C(Z, shell, @error);
    process_error(error);
    result := rv
end;

function AugerRate_C(Z:longint; auger_trans:longint; error:PPxrl_error):double;cdecl;external External_library name 'AugerRate';

function AugerRate(Z:longint; auger_trans:longint):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := AugerRate_C(Z, auger_trans, @error);
    process_error(error);
    result := rv
end;

function AugerYield_C(Z:longint; shell:longint; error:PPxrl_error):double;cdecl;external External_library name 'AugerYield';

function AugerYield(Z:longint; shell:longint):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := AugerYield_C(Z, shell, @error);
    process_error(error);
    result := rv
end;

function CS_FluorLine_Kissel_C(Z:longint; line:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_FluorLine_Kissel';

function CS_FluorLine_Kissel(Z:longint; line:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_FluorLine_Kissel_C(Z, line, E, @error);
    process_error(error);
    result := rv
end;

function CSb_FluorLine_Kissel_C(Z:longint; line:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_FluorLine_Kissel';

function CSb_FluorLine_Kissel(Z:longint; line:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_FluorLine_Kissel_C(Z, line, E, @error);
    process_error(error);
    result := rv
end;

function CS_FluorLine_Kissel_Cascade_C(Z:longint; line:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_FluorLine_Kissel_Cascade';

function CS_FluorLine_Kissel_Cascade(Z:longint; line:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_FluorLine_Kissel_Cascade_C(Z, line, E, @error);
    process_error(error);
    result := rv
end;

function CSb_FluorLine_Kissel_Cascade_C(Z:longint; line:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_FluorLine_Kissel_Cascade';

function CSb_FluorLine_Kissel_Cascade(Z:longint; line:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_FluorLine_Kissel_Cascade_C(Z, line, E, @error);
    process_error(error);
    result := rv
end;

function CS_FluorLine_Kissel_Nonradiative_Cascade_C(Z:longint; line:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_FluorLine_Kissel_Nonradiative_Cascade';

function CS_FluorLine_Kissel_Nonradiative_Cascade(Z:longint; line:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_FluorLine_Kissel_Nonradiative_Cascade_C(Z, line, E, @error);
    process_error(error);
    result := rv
end;

function CSb_FluorLine_Kissel_Nonradiative_Cascade_C(Z:longint; line:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_FluorLine_Kissel_Nonradiative_Cascade';

function CSb_FluorLine_Kissel_Nonradiative_Cascade(Z:longint; line:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_FluorLine_Kissel_Nonradiative_Cascade_C(Z, line, E, @error);
    process_error(error);
    result := rv
end;

function CS_FluorLine_Kissel_Radiative_Cascade_C(Z:longint; line:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_FluorLine_Kissel_Radiative_Cascade';

function CS_FluorLine_Kissel_Radiative_Cascade(Z:longint; line:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_FluorLine_Kissel_Radiative_Cascade_C(Z, line, E, @error);
    process_error(error);
    result := rv
end;

function CSb_FluorLine_Kissel_Radiative_Cascade_C(Z:longint; line:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_FluorLine_Kissel_Radiative_Cascade';

function CSb_FluorLine_Kissel_Radiative_Cascade(Z:longint; line:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_FluorLine_Kissel_Radiative_Cascade_C(Z, line, E, @error);
    process_error(error);
    result := rv
end;

function CS_FluorLine_Kissel_no_Cascade_C(Z:longint; line:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_FluorLine_Kissel_no_Cascade';

function CS_FluorLine_Kissel_no_Cascade(Z:longint; line:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_FluorLine_Kissel_no_Cascade_C(Z, line, E, @error);
    process_error(error);
    result := rv
end;

function CSb_FluorLine_Kissel_no_Cascade_C(Z:longint; line:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_FluorLine_Kissel_no_Cascade';

function CSb_FluorLine_Kissel_no_Cascade(Z:longint; line:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_FluorLine_Kissel_no_Cascade_C(Z, line, E, @error);
    process_error(error);
    result := rv
end;

function CS_FluorShell_Kissel_C(Z:longint; shell:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_FluorShell_Kissel';

function CS_FluorShell_Kissel(Z:longint; shell:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_FluorShell_Kissel_C(Z, shell, E, @error);
    process_error(error);
    result := rv
end;

function CSb_FluorShell_Kissel_C(Z:longint; shell:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_FluorShell_Kissel';

function CSb_FluorShell_Kissel(Z:longint; shell:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_FluorShell_Kissel_C(Z, shell, E, @error);
    process_error(error);
    result := rv
end;

function CS_FluorShell_Kissel_Cascade_C(Z:longint; shell:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_FluorShell_Kissel_Cascade';

function CS_FluorShell_Kissel_Cascade(Z:longint; shell:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_FluorShell_Kissel_Cascade_C(Z, shell, E, @error);
    process_error(error);
    result := rv
end;

function CSb_FluorShell_Kissel_Cascade_C(Z:longint; shell:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_FluorShell_Kissel_Cascade';

function CSb_FluorShell_Kissel_Cascade(Z:longint; shell:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_FluorShell_Kissel_Cascade_C(Z, shell, E, @error);
    process_error(error);
    result := rv
end;

function CS_FluorShell_Kissel_Nonradiative_Cascade_C(Z:longint; shell:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_FluorShell_Kissel_Nonradiative_Cascade';

function CS_FluorShell_Kissel_Nonradiative_Cascade(Z:longint; shell:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_FluorShell_Kissel_Nonradiative_Cascade_C(Z, shell, E, @error);
    process_error(error);
    result := rv
end;

function CSb_FluorShell_Kissel_Nonradiative_Cascade_C(Z:longint; shell:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_FluorShell_Kissel_Nonradiative_Cascade';

function CSb_FluorShell_Kissel_Nonradiative_Cascade(Z:longint; shell:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_FluorShell_Kissel_Nonradiative_Cascade_C(Z, shell, E, @error);
    process_error(error);
    result := rv
end;

function CS_FluorShell_Kissel_Radiative_Cascade_C(Z:longint; shell:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_FluorShell_Kissel_Radiative_Cascade';

function CS_FluorShell_Kissel_Radiative_Cascade(Z:longint; shell:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_FluorShell_Kissel_Radiative_Cascade_C(Z, shell, E, @error);
    process_error(error);
    result := rv
end;

function CSb_FluorShell_Kissel_Radiative_Cascade_C(Z:longint; shell:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_FluorShell_Kissel_Radiative_Cascade';

function CSb_FluorShell_Kissel_Radiative_Cascade(Z:longint; shell:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_FluorShell_Kissel_Radiative_Cascade_C(Z, shell, E, @error);
    process_error(error);
    result := rv
end;

function CS_FluorShell_Kissel_no_Cascade_C(Z:longint; shell:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_FluorShell_Kissel_no_Cascade';

function CS_FluorShell_Kissel_no_Cascade(Z:longint; shell:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CS_FluorShell_Kissel_no_Cascade_C(Z, shell, E, @error);
    process_error(error);
    result := rv
end;

function CSb_FluorShell_Kissel_no_Cascade_C(Z:longint; shell:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_FluorShell_Kissel_no_Cascade';

function CSb_FluorShell_Kissel_no_Cascade(Z:longint; shell:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := CSb_FluorShell_Kissel_no_Cascade_C(Z, shell, E, @error);
    process_error(error);
    result := rv
end;

function PL1_pure_kissel_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'PL1_pure_kissel';

function PL1_pure_kissel(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PL1_pure_kissel_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function PL1_rad_cascade_kissel_C(Z:longint; E:double; PK:double; error:PPxrl_error):double;cdecl;external External_library name 'PL1_rad_cascade_kissel';

function PL1_rad_cascade_kissel(Z:longint; E:double; PK:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PL1_rad_cascade_kissel_C(Z, E, PK, @error);
    process_error(error);
    result := rv
end;

function PL1_auger_cascade_kissel_C(Z:longint; E:double; PK:double; error:PPxrl_error):double;cdecl;external External_library name 'PL1_auger_cascade_kissel';

function PL1_auger_cascade_kissel(Z:longint; E:double; PK:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PL1_auger_cascade_kissel_C(Z, E, PK, @error);
    process_error(error);
    result := rv
end;

function PL1_full_cascade_kissel_C(Z:longint; E:double; PK:double; error:PPxrl_error):double;cdecl;external External_library name 'PL1_full_cascade_kissel';

function PL1_full_cascade_kissel(Z:longint; E:double; PK:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PL1_full_cascade_kissel_C(Z, E, PK, @error);
    process_error(error);
    result := rv
end;

function PL2_pure_kissel_C(Z:longint; E:double; PL1:double; error:PPxrl_error):double;cdecl;external External_library name 'PL2_pure_kissel';

function PL2_pure_kissel(Z:longint; E:double; PL1:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PL2_pure_kissel_C(Z, E, PL1, @error);
    process_error(error);
    result := rv
end;

function PL2_rad_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; error:PPxrl_error):double;cdecl;external External_library name 'PL2_rad_cascade_kissel';

function PL2_rad_cascade_kissel(Z:longint; E:double; PK:double; PL1:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PL2_rad_cascade_kissel_C(Z, E, PK, PL1, @error);
    process_error(error);
    result := rv
end;

function PL2_auger_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; error:PPxrl_error):double;cdecl;external External_library name 'PL2_auger_cascade_kissel';

function PL2_auger_cascade_kissel(Z:longint; E:double; PK:double; PL1:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PL2_auger_cascade_kissel_C(Z, E, PK, PL1, @error);
    process_error(error);
    result := rv
end;

function PL2_full_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; error:PPxrl_error):double;cdecl;external External_library name 'PL2_full_cascade_kissel';

function PL2_full_cascade_kissel(Z:longint; E:double; PK:double; PL1:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PL2_full_cascade_kissel_C(Z, E, PK, PL1, @error);
    process_error(error);
    result := rv
end;

function PL3_pure_kissel_C(Z:longint; E:double; PL1:double; PL2:double; error:PPxrl_error):double;cdecl;external External_library name 'PL3_pure_kissel';

function PL3_pure_kissel(Z:longint; E:double; PL1:double; PL2:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PL3_pure_kissel_C(Z, E, PL1, PL2, @error);
    process_error(error);
    result := rv
end;

function PL3_rad_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; error:PPxrl_error):double;cdecl;external External_library name 'PL3_rad_cascade_kissel';

function PL3_rad_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PL3_rad_cascade_kissel_C(Z, E, PK, PL1, PL2, @error);
    process_error(error);
    result := rv
end;

function PL3_auger_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; error:PPxrl_error):double;cdecl;external External_library name 'PL3_auger_cascade_kissel';

function PL3_auger_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PL3_auger_cascade_kissel_C(Z, E, PK, PL1, PL2, @error);
    process_error(error);
    result := rv
end;

function PL3_full_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; error:PPxrl_error):double;cdecl;external External_library name 'PL3_full_cascade_kissel';

function PL3_full_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PL3_full_cascade_kissel_C(Z, E, PK, PL1, PL2, @error);
    process_error(error);
    result := rv
end;

function PM1_pure_kissel_C(Z:longint; E:double; error:PPxrl_error):double;cdecl;external External_library name 'PM1_pure_kissel';

function PM1_pure_kissel(Z:longint; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM1_pure_kissel_C(Z, E, @error);
    process_error(error);
    result := rv
end;

function PM1_rad_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; error:PPxrl_error):double;cdecl;external External_library name 'PM1_rad_cascade_kissel';

function PM1_rad_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM1_rad_cascade_kissel_C(Z, E, PK, PL1, PL2, PL3, @error);
    process_error(error);
    result := rv
end;

function PM1_auger_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; error:PPxrl_error):double;cdecl;external External_library name 'PM1_auger_cascade_kissel';

function PM1_auger_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM1_auger_cascade_kissel_C(Z, E, PK, PL1, PL2, PL3, @error);
    process_error(error);
    result := rv
end;

function PM1_full_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; error:PPxrl_error):double;cdecl;external External_library name 'PM1_full_cascade_kissel';

function PM1_full_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM1_full_cascade_kissel_C(Z, E, PK, PL1, PL2, PL3, @error);
    process_error(error);
    result := rv
end;

function PM2_pure_kissel_C(Z:longint; E:double; PM1:double; error:PPxrl_error):double;cdecl;external External_library name 'PM2_pure_kissel';

function PM2_pure_kissel(Z:longint; E:double; PM1:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM2_pure_kissel_C(Z, E, PM1, @error);
    process_error(error);
    result := rv
end;

function PM2_rad_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; error:PPxrl_error):double;cdecl;external External_library name 'PM2_rad_cascade_kissel';

function PM2_rad_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM2_rad_cascade_kissel_C(Z, E, PK, PL1, PL2, PL3, PM1, @error);
    process_error(error);
    result := rv
end;

function PM2_auger_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; error:PPxrl_error):double;cdecl;external External_library name 'PM2_auger_cascade_kissel';

function PM2_auger_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM2_auger_cascade_kissel_C(Z, E, PK, PL1, PL2, PL3, PM1, @error);
    process_error(error);
    result := rv
end;

function PM2_full_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; error:PPxrl_error):double;cdecl;external External_library name 'PM2_full_cascade_kissel';

function PM2_full_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM2_full_cascade_kissel_C(Z, E, PK, PL1, PL2, PL3, PM1, @error);
    process_error(error);
    result := rv
end;

function PM3_pure_kissel_C(Z:longint; E:double; PM1:double; PM2:double; error:PPxrl_error):double;cdecl;external External_library name 'PM3_pure_kissel';

function PM3_pure_kissel(Z:longint; E:double; PM1:double; PM2:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM3_pure_kissel_C(Z, E, PM1, PM2, @error);
    process_error(error);
    result := rv
end;

function PM3_rad_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double; error:PPxrl_error):double;cdecl;external External_library name 'PM3_rad_cascade_kissel';

function PM3_rad_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM3_rad_cascade_kissel_C(Z, E, PK, PL1, PL2, PL3, PM1, PM2, @error);
    process_error(error);
    result := rv
end;

function PM3_auger_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double; error:PPxrl_error):double;cdecl;external External_library name 'PM3_auger_cascade_kissel';

function PM3_auger_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM3_auger_cascade_kissel_C(Z, E, PK, PL1, PL2, PL3, PM1, PM2, @error);
    process_error(error);
    result := rv
end;

function PM3_full_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double; error:PPxrl_error):double;cdecl;external External_library name 'PM3_full_cascade_kissel';

function PM3_full_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM3_full_cascade_kissel_C(Z, E, PK, PL1, PL2, PL3, PM1, PM2, @error);
    process_error(error);
    result := rv
end;

function PM4_pure_kissel_C(Z:longint; E:double; PM1:double; PM2:double; PM3:double; error:PPxrl_error):double;cdecl;external External_library name 'PM4_pure_kissel';

function PM4_pure_kissel(Z:longint; E:double; PM1:double; PM2:double; PM3:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM4_pure_kissel_C(Z, E, PM1, PM2, PM3, @error);
    process_error(error);
    result := rv
end;

function PM4_rad_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double; PM3:double; error:PPxrl_error):double;cdecl;external External_library name 'PM4_rad_cascade_kissel';

function PM4_rad_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double; PM3:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM4_rad_cascade_kissel_C(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, @error);
    process_error(error);
    result := rv
end;

function PM4_auger_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double; PM3:double; error:PPxrl_error):double;cdecl;external External_library name 'PM4_auger_cascade_kissel';

function PM4_auger_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double; PM3:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM4_auger_cascade_kissel_C(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, @error);
    process_error(error);
    result := rv
end;

function PM4_full_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double; PM3:double; error:PPxrl_error):double;cdecl;external External_library name 'PM4_full_cascade_kissel';

function PM4_full_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double; PM3:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM4_full_cascade_kissel_C(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, @error);
    process_error(error);
    result := rv
end;

function PM5_pure_kissel_C(Z:longint; E:double; PM1:double; PM2:double; PM3:double; PM4:double; error:PPxrl_error):double;cdecl;external External_library name 'PM5_pure_kissel';

function PM5_pure_kissel(Z:longint; E:double; PM1:double; PM2:double; PM3:double; PM4:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM5_pure_kissel_C(Z, E, PM1, PM2, PM3, PM4, @error);
    process_error(error);
    result := rv
end;

function PM5_rad_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double; PM3:double; PM4:double; error:PPxrl_error):double;cdecl;external External_library name 'PM5_rad_cascade_kissel';

function PM5_rad_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double; PM3:double; PM4:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM5_rad_cascade_kissel_C(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, @error);
    process_error(error);
    result := rv
end;

function PM5_auger_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double; PM3:double; PM4:double; error:PPxrl_error):double;cdecl;external External_library name 'PM5_auger_cascade_kissel';

function PM5_auger_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double; PM3:double; PM4:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM5_auger_cascade_kissel_C(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, @error);
    process_error(error);
    result := rv
end;

function PM5_full_cascade_kissel_C(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double; PM3:double; PM4:double; error:PPxrl_error):double;cdecl;external External_library name 'PM5_full_cascade_kissel';

function PM5_full_cascade_kissel(Z:longint; E:double; PK:double; PL1:double; PL2:double; PL3:double; PM1:double; PM2:double; PM3:double; PM4:double):double;

var
    error: Pxrl_error;
    rv: double;
    
begin
    error := nil;
    
    rv := PM5_full_cascade_kissel_C(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, @error);
    process_error(error);
    result := rv
end;

function CS_Total_CP_C(compound:PAnsiChar; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_Total_CP';

function CS_Total_CP(compound:string; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := CS_Total_CP_C(compound_c, E, @error);
    process_error(error);
    result := rv
end;

function CS_Photo_CP_C(compound:PAnsiChar; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_Photo_CP';

function CS_Photo_CP(compound:string; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := CS_Photo_CP_C(compound_c, E, @error);
    process_error(error);
    result := rv
end;

function CS_Rayl_CP_C(compound:PAnsiChar; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_Rayl_CP';

function CS_Rayl_CP(compound:string; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := CS_Rayl_CP_C(compound_c, E, @error);
    process_error(error);
    result := rv
end;

function CS_Compt_CP_C(compound:PAnsiChar; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_Compt_CP';

function CS_Compt_CP(compound:string; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := CS_Compt_CP_C(compound_c, E, @error);
    process_error(error);
    result := rv
end;

function CS_Energy_CP_C(compound:PAnsiChar; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_Energy_CP';

function CS_Energy_CP(compound:string; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := CS_Energy_CP_C(compound_c, E, @error);
    process_error(error);
    result := rv
end;

function CSb_Total_CP_C(compound:PAnsiChar; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_Total_CP';

function CSb_Total_CP(compound:string; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := CSb_Total_CP_C(compound_c, E, @error);
    process_error(error);
    result := rv
end;

function CSb_Photo_CP_C(compound:PAnsiChar; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_Photo_CP';

function CSb_Photo_CP(compound:string; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := CSb_Photo_CP_C(compound_c, E, @error);
    process_error(error);
    result := rv
end;

function CSb_Rayl_CP_C(compound:PAnsiChar; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_Rayl_CP';

function CSb_Rayl_CP(compound:string; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := CSb_Rayl_CP_C(compound_c, E, @error);
    process_error(error);
    result := rv
end;

function CSb_Compt_CP_C(compound:PAnsiChar; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_Compt_CP';

function CSb_Compt_CP(compound:string; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := CSb_Compt_CP_C(compound_c, E, @error);
    process_error(error);
    result := rv
end;

function DCS_Rayl_CP_C(compound:PAnsiChar; E:double; theta:double; error:PPxrl_error):double;cdecl;external External_library name 'DCS_Rayl_CP';

function DCS_Rayl_CP(compound:string; E:double; theta:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := DCS_Rayl_CP_C(compound_c, E, theta, @error);
    process_error(error);
    result := rv
end;

function DCS_Compt_CP_C(compound:PAnsiChar; E:double; theta:double; error:PPxrl_error):double;cdecl;external External_library name 'DCS_Compt_CP';

function DCS_Compt_CP(compound:string; E:double; theta:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := DCS_Compt_CP_C(compound_c, E, theta, @error);
    process_error(error);
    result := rv
end;

function DCSb_Rayl_CP_C(compound:PAnsiChar; E:double; theta:double; error:PPxrl_error):double;cdecl;external External_library name 'DCSb_Rayl_CP';

function DCSb_Rayl_CP(compound:string; E:double; theta:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := DCSb_Rayl_CP_C(compound_c, E, theta, @error);
    process_error(error);
    result := rv
end;

function DCSb_Compt_CP_C(compound:PAnsiChar; E:double; theta:double; error:PPxrl_error):double;cdecl;external External_library name 'DCSb_Compt_CP';

function DCSb_Compt_CP(compound:string; E:double; theta:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := DCSb_Compt_CP_C(compound_c, E, theta, @error);
    process_error(error);
    result := rv
end;

function DCSP_Rayl_CP_C(compound:PAnsiChar; E:double; theta:double; phi:double; error:PPxrl_error):double;cdecl;external External_library name 'DCSP_Rayl_CP';

function DCSP_Rayl_CP(compound:string; E:double; theta:double; phi:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := DCSP_Rayl_CP_C(compound_c, E, theta, phi, @error);
    process_error(error);
    result := rv
end;

function DCSP_Compt_CP_C(compound:PAnsiChar; E:double; theta:double; phi:double; error:PPxrl_error):double;cdecl;external External_library name 'DCSP_Compt_CP';

function DCSP_Compt_CP(compound:string; E:double; theta:double; phi:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := DCSP_Compt_CP_C(compound_c, E, theta, phi, @error);
    process_error(error);
    result := rv
end;

function DCSPb_Rayl_CP_C(compound:PAnsiChar; E:double; theta:double; phi:double; error:PPxrl_error):double;cdecl;external External_library name 'DCSPb_Rayl_CP';

function DCSPb_Rayl_CP(compound:string; E:double; theta:double; phi:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := DCSPb_Rayl_CP_C(compound_c, E, theta, phi, @error);
    process_error(error);
    result := rv
end;

function DCSPb_Compt_CP_C(compound:PAnsiChar; E:double; theta:double; phi:double; error:PPxrl_error):double;cdecl;external External_library name 'DCSPb_Compt_CP';

function DCSPb_Compt_CP(compound:string; E:double; theta:double; phi:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := DCSPb_Compt_CP_C(compound_c, E, theta, phi, @error);
    process_error(error);
    result := rv
end;

function CS_Photo_Total_CP_C(compound:PAnsiChar; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_Photo_Total_CP';

function CS_Photo_Total_CP(compound:string; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := CS_Photo_Total_CP_C(compound_c, E, @error);
    process_error(error);
    result := rv
end;

function CS_Total_Kissel_CP_C(compound:PAnsiChar; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CS_Total_Kissel_CP';

function CS_Total_Kissel_CP(compound:string; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := CS_Total_Kissel_CP_C(compound_c, E, @error);
    process_error(error);
    result := rv
end;

function CSb_Photo_Total_CP_C(compound:PAnsiChar; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_Photo_Total_CP';

function CSb_Photo_Total_CP(compound:string; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := CSb_Photo_Total_CP_C(compound_c, E, @error);
    process_error(error);
    result := rv
end;

function CSb_Total_Kissel_CP_C(compound:PAnsiChar; E:double; error:PPxrl_error):double;cdecl;external External_library name 'CSb_Total_Kissel_CP';

function CSb_Total_Kissel_CP(compound:string; E:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := CSb_Total_Kissel_CP_C(compound_c, E, @error);
    process_error(error);
    result := rv
end;

function Refractive_Index_Re_C(compound:PAnsiChar; E:double; density:double; error:PPxrl_error):double;cdecl;external External_library name 'Refractive_Index_Re';

function Refractive_Index_Re(compound:string; E:double; density:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := Refractive_Index_Re_C(compound_c, E, density, @error);
    process_error(error);
    result := rv
end;

function Refractive_Index_Im_C(compound:PAnsiChar; E:double; density:double; error:PPxrl_error):double;cdecl;external External_library name 'Refractive_Index_Im';

function Refractive_Index_Im(compound:string; E:double; density:double):double;

var
    error: Pxrl_error;
    rv: double;
    compound_c:PAnsiChar;

begin
    error := nil;
    compound_c := PAnsichar(AnsiString(compound));

    rv := Refractive_Index_Im_C(compound_c, E, density, @error);
    process_error(error);
    result := rv
end;

