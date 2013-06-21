  ;--------------------------------
 
  ; Macros for mutually exclusive section selection
  ; Written by Tim Gallagher
  ;
  ; See one-section.nsi for an example of usage
 
  !include LogicLib.nsh
 
  !macro StartRadioButtons _d
    !verbose push
    !verbose ${LOGICLIB_VERBOSITY}
 
    !ifndef _CheckedButton${_d}
      var /GLOBAL _CheckedButton${_d}
      !define _CheckedButton${_d}
    !endif
 
    ${IfThen} $_CheckedButton${_d} == "" ${|} StrCpy $_CheckedButton${_d} `${_d}` ${|}
    !define CheckedButton `$_CheckedButton${_d}`
 
    Push $0
    Push $1
 
    SectionGetFlags `${CheckedButton}` $0
    IntOp $0 $0 & ${SECTION_OFF}
    SectionSetFlags `${CheckedButton}` $0
 
    StrCpy $1 `${CheckedButton}`
 
    !verbose pop
  !macroend
 
  ; A radio button
  !macro RadioButton _s
    !verbose push
    !verbose ${LOGICLIB_VERBOSITY}
 
    ${If} ${SectionIsSelected} `${_s}`
      StrCpy `${CheckedButton}` `${_s}`
    ${EndIf}
 
    !verbose pop
  !macroend
 
  ; Ends the radio button block
  !macro EndRadioButtons
    !verbose push
    !verbose ${LOGICLIB_VERBOSITY}
 
    ${If} $1 == `${CheckedButton}`                        ; selection hasn't changed
      SectionGetFlags `${CheckedButton}` $0
      IntOp $0 $0 | ${SF_SELECTED}
      SectionSetFlags `${CheckedButton}` $0
    ${EndIf}
 
    Pop $1
    Pop $0
 
    !undef CheckedButton
    !verbose pop
  !macroend
 
  !macro SectionRadioButtons _d _e
    Push $2
    Push $3
    Push $4
    Push $5
    Push $6
    Push $7
 
    !insertmacro StartRadioButtons ${_d}
 
    StrCpy $3 0
    StrLen $5 `${_e}`
    ${For} $2 0 $5
      StrCpy $4 `${_e}` 1 $2
      ${If} $4 == ","
        IntOp $3 $3 + 1
      ${EndIf}
    ${Next}
 
    StrCpy $7 `${_e}`
    StrCpy $4 0
    ${Do}
 
      StrLen $5 `$7`
      ${For} $2 0 $5
        StrCpy $6 `$7` 1 $2
        ${If} $6 == ","
          StrCpy $6 `$7` $2
          IntOp $2 $2 + 1
          StrCpy $7 `$7` "" $2
          IntOp $4 $4 + 1
          ${Break}
        ${EndIf}
      ${Next}
 
      ${If} $4 <= $3
      ${AndIf} $6 != ""
        !insertmacro RadioButton $6
      ${AndIf} $4 == $3
        !insertmacro RadioButton $7
      ${EndIf}
 
    ${LoopUntil} $4 >= $3
 
    !insertmacro EndRadioButtons
 
    Pop $7
    Pop $6
    Pop $5
    Pop $4
    Pop $3
    Pop $2
  !macroend
 
  !macro RadioGetChecked _d _e
    Push $2
    Push $3
    Push $4
    Push $5
    Push $6
    Push $7
 
    !ifndef _CheckedButton${_d}
      var /GLOBAL _CheckedButton${_d}
      !define _CheckedButton${_d}
    !endif
 
    StrCpy $3 0
    StrLen $5 `${_e}`
    ${For} $2 0 $5
      StrCpy $4 `${_e}` 1 $2
      ${If} $4 == ","
        IntOp $3 $3 + 1
      ${EndIf}
    ${Next}
 
    StrCpy $7 `${_e}`
    StrCpy $4 0
    ${Do}
 
      StrLen $5 `$7`
      ${For} $2 0 $5
        StrCpy $6 `$7` 1 $2
        ${If} $6 == ","
          StrCpy $6 `$7` $2
          IntOp $2 $2 + 1
          StrCpy $7 `$7` "" $2
          IntOp $4 $4 + 1
          ${Break}
        ${EndIf}
      ${Next}
 
      ${If} $4 <= $3
      ${AndIf} $6 != ""
 
        ${If} ${SectionIsSelected} $6
          StrCpy $_CheckedButton${_d} "$6"
          ${ExitDo}
        ${EndIf}
 
        ${If} $4 == $3
        ${AndIf} ${SectionIsSelected} $7
          StrCpy $_CheckedButton${_d} "$7"
        ${EndIf}
      ${EndIf}
 
    ${LoopUntil} $4 >= $3
 
    Pop $7
    Pop $6
    Pop $5
    Pop $4
    Pop $3
    Pop $2
  !macroend
