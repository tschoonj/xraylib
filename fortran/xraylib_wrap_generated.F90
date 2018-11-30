! This file has been generated automatically using generate-code.py


FUNCTION AtomicWeight(Z, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION AtomicWeightC(Z, error) &
            BIND(C,NAME='AtomicWeight')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE) :: AtomicWeightC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION AtomicWeightC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = AtomicWeightC(Z, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION AtomicWeight


FUNCTION ElementDensity(Z, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION ElementDensityC(Z, error) &
            BIND(C,NAME='ElementDensity')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE) :: ElementDensityC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION ElementDensityC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = ElementDensityC(Z, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION ElementDensity


FUNCTION CS_Total(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_TotalC(Z, E, error) &
            BIND(C,NAME='CS_Total')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_TotalC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_TotalC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CS_TotalC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_Total


FUNCTION CS_Photo(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_PhotoC(Z, E, error) &
            BIND(C,NAME='CS_Photo')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_PhotoC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_PhotoC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CS_PhotoC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_Photo


FUNCTION CS_Rayl(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_RaylC(Z, E, error) &
            BIND(C,NAME='CS_Rayl')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_RaylC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_RaylC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CS_RaylC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_Rayl


FUNCTION CS_Compt(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_ComptC(Z, E, error) &
            BIND(C,NAME='CS_Compt')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_ComptC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_ComptC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CS_ComptC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_Compt


FUNCTION CS_Energy(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_EnergyC(Z, E, error) &
            BIND(C,NAME='CS_Energy')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_EnergyC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_EnergyC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CS_EnergyC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_Energy


FUNCTION CS_KN(E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_KNC(E, error) &
            BIND(C,NAME='CS_KN')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_KNC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_KNC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CS_KNC(E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_KN


FUNCTION CSb_Total(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_TotalC(Z, E, error) &
            BIND(C,NAME='CSb_Total')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_TotalC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_TotalC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CSb_TotalC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_Total


FUNCTION CSb_Photo(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_PhotoC(Z, E, error) &
            BIND(C,NAME='CSb_Photo')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_PhotoC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_PhotoC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CSb_PhotoC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_Photo


FUNCTION CSb_Rayl(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_RaylC(Z, E, error) &
            BIND(C,NAME='CSb_Rayl')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_RaylC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_RaylC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CSb_RaylC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_Rayl


FUNCTION CSb_Compt(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_ComptC(Z, E, error) &
            BIND(C,NAME='CSb_Compt')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_ComptC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_ComptC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CSb_ComptC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_Compt


FUNCTION DCS_Thoms(theta, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCS_ThomsC(theta, error) &
            BIND(C,NAME='DCS_Thoms')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE) :: DCS_ThomsC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCS_ThomsC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = DCS_ThomsC(theta, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCS_Thoms


FUNCTION DCS_KN(E, theta, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCS_KNC(E, theta, error) &
            BIND(C,NAME='DCS_KN')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE) :: DCS_KNC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCS_KNC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = DCS_KNC(E, theta, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCS_KN


FUNCTION DCS_Rayl(Z, E, theta, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCS_RaylC(Z, E, theta, error) &
            BIND(C,NAME='DCS_Rayl')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE) :: DCS_RaylC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCS_RaylC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = DCS_RaylC(Z, E, theta, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCS_Rayl


FUNCTION DCS_Compt(Z, E, theta, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCS_ComptC(Z, E, theta, error) &
            BIND(C,NAME='DCS_Compt')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE) :: DCS_ComptC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCS_ComptC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = DCS_ComptC(Z, E, theta, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCS_Compt


FUNCTION DCSb_Rayl(Z, E, theta, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCSb_RaylC(Z, E, theta, error) &
            BIND(C,NAME='DCSb_Rayl')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE) :: DCSb_RaylC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCSb_RaylC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = DCSb_RaylC(Z, E, theta, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCSb_Rayl


FUNCTION DCSb_Compt(Z, E, theta, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCSb_ComptC(Z, E, theta, error) &
            BIND(C,NAME='DCSb_Compt')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE) :: DCSb_ComptC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCSb_ComptC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = DCSb_ComptC(Z, E, theta, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCSb_Compt


FUNCTION DCSP_Thoms(theta, phi, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE), INTENT(IN) :: phi
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCSP_ThomsC(theta, phi, error) &
            BIND(C,NAME='DCSP_Thoms')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE), INTENT(IN), VALUE :: phi
                REAL (C_DOUBLE) :: DCSP_ThomsC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCSP_ThomsC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = DCSP_ThomsC(theta, phi, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCSP_Thoms


FUNCTION DCSP_KN(E, theta, phi, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE), INTENT(IN) :: phi
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCSP_KNC(E, theta, phi, error) &
            BIND(C,NAME='DCSP_KN')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE), INTENT(IN), VALUE :: phi
                REAL (C_DOUBLE) :: DCSP_KNC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCSP_KNC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = DCSP_KNC(E, theta, phi, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCSP_KN


FUNCTION DCSP_Rayl(Z, E, theta, phi, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE), INTENT(IN) :: phi
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCSP_RaylC(Z, E, theta, phi, error) &
            BIND(C,NAME='DCSP_Rayl')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE), INTENT(IN), VALUE :: phi
                REAL (C_DOUBLE) :: DCSP_RaylC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCSP_RaylC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = DCSP_RaylC(Z, E, theta, phi, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCSP_Rayl


FUNCTION DCSP_Compt(Z, E, theta, phi, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE), INTENT(IN) :: phi
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCSP_ComptC(Z, E, theta, phi, error) &
            BIND(C,NAME='DCSP_Compt')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE), INTENT(IN), VALUE :: phi
                REAL (C_DOUBLE) :: DCSP_ComptC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCSP_ComptC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = DCSP_ComptC(Z, E, theta, phi, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCSP_Compt


FUNCTION DCSPb_Rayl(Z, E, theta, phi, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE), INTENT(IN) :: phi
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCSPb_RaylC(Z, E, theta, phi, error) &
            BIND(C,NAME='DCSPb_Rayl')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE), INTENT(IN), VALUE :: phi
                REAL (C_DOUBLE) :: DCSPb_RaylC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCSPb_RaylC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = DCSPb_RaylC(Z, E, theta, phi, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCSPb_Rayl


FUNCTION DCSPb_Compt(Z, E, theta, phi, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE), INTENT(IN) :: phi
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCSPb_ComptC(Z, E, theta, phi, error) &
            BIND(C,NAME='DCSPb_Compt')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE), INTENT(IN), VALUE :: phi
                REAL (C_DOUBLE) :: DCSPb_ComptC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCSPb_ComptC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = DCSPb_ComptC(Z, E, theta, phi, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCSPb_Compt


FUNCTION FF_Rayl(Z, q, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: q
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION FF_RaylC(Z, q, error) &
            BIND(C,NAME='FF_Rayl')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: q
                REAL (C_DOUBLE) :: FF_RaylC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION FF_RaylC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = FF_RaylC(Z, q, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION FF_Rayl


FUNCTION SF_Compt(Z, q, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: q
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION SF_ComptC(Z, q, error) &
            BIND(C,NAME='SF_Compt')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: q
                REAL (C_DOUBLE) :: SF_ComptC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION SF_ComptC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = SF_ComptC(Z, q, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION SF_Compt


FUNCTION MomentTransf(E, theta, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION MomentTransfC(E, theta, error) &
            BIND(C,NAME='MomentTransf')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE) :: MomentTransfC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION MomentTransfC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = MomentTransfC(E, theta, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION MomentTransf


FUNCTION LineEnergy(Z, line, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: line
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION LineEnergyC(Z, line, error) &
            BIND(C,NAME='LineEnergy')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: line
                REAL (C_DOUBLE) :: LineEnergyC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION LineEnergyC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = LineEnergyC(Z, line, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION LineEnergy


FUNCTION FluorYield(Z, shell, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: shell
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION FluorYieldC(Z, shell, error) &
            BIND(C,NAME='FluorYield')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: shell
                REAL (C_DOUBLE) :: FluorYieldC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION FluorYieldC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = FluorYieldC(Z, shell, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION FluorYield


FUNCTION CosKronTransProb(Z, trans, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: trans
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CosKronTransProbC(Z, trans, error) &
            BIND(C,NAME='CosKronTransProb')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: trans
                REAL (C_DOUBLE) :: CosKronTransProbC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CosKronTransProbC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CosKronTransProbC(Z, trans, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CosKronTransProb


FUNCTION EdgeEnergy(Z, shell, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: shell
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION EdgeEnergyC(Z, shell, error) &
            BIND(C,NAME='EdgeEnergy')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: shell
                REAL (C_DOUBLE) :: EdgeEnergyC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION EdgeEnergyC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = EdgeEnergyC(Z, shell, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION EdgeEnergy


FUNCTION JumpFactor(Z, shell, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: shell
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION JumpFactorC(Z, shell, error) &
            BIND(C,NAME='JumpFactor')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: shell
                REAL (C_DOUBLE) :: JumpFactorC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION JumpFactorC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = JumpFactorC(Z, shell, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION JumpFactor


FUNCTION CS_FluorLine(Z, line, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: line
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_FluorLineC(Z, line, E, error) &
            BIND(C,NAME='CS_FluorLine')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: line
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_FluorLineC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_FluorLineC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CS_FluorLineC(Z, line, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_FluorLine


FUNCTION CSb_FluorLine(Z, line, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: line
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_FluorLineC(Z, line, E, error) &
            BIND(C,NAME='CSb_FluorLine')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: line
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_FluorLineC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_FluorLineC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CSb_FluorLineC(Z, line, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_FluorLine


FUNCTION RadRate(Z, line, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: line
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION RadRateC(Z, line, error) &
            BIND(C,NAME='RadRate')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: line
                REAL (C_DOUBLE) :: RadRateC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION RadRateC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = RadRateC(Z, line, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION RadRate


FUNCTION ComptonEnergy(E0, theta, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        REAL (C_DOUBLE), INTENT(IN) :: E0
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION ComptonEnergyC(E0, theta, error) &
            BIND(C,NAME='ComptonEnergy')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E0
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE) :: ComptonEnergyC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION ComptonEnergyC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = ComptonEnergyC(E0, theta, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION ComptonEnergy


FUNCTION Fi(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION FiC(Z, E, error) &
            BIND(C,NAME='Fi')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: FiC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION FiC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = FiC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION Fi


FUNCTION Fii(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION FiiC(Z, E, error) &
            BIND(C,NAME='Fii')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: FiiC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION FiiC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = FiiC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION Fii


FUNCTION CS_Photo_Total(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_Photo_TotalC(Z, E, error) &
            BIND(C,NAME='CS_Photo_Total')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_Photo_TotalC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_Photo_TotalC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CS_Photo_TotalC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_Photo_Total


FUNCTION CSb_Photo_Total(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_Photo_TotalC(Z, E, error) &
            BIND(C,NAME='CSb_Photo_Total')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_Photo_TotalC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_Photo_TotalC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CSb_Photo_TotalC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_Photo_Total


FUNCTION CS_Photo_Partial(Z, shell, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: shell
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_Photo_PartialC(Z, shell, E, error) &
            BIND(C,NAME='CS_Photo_Partial')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: shell
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_Photo_PartialC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_Photo_PartialC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CS_Photo_PartialC(Z, shell, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_Photo_Partial


FUNCTION CSb_Photo_Partial(Z, shell, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: shell
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_Photo_PartialC(Z, shell, E, error) &
            BIND(C,NAME='CSb_Photo_Partial')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: shell
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_Photo_PartialC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_Photo_PartialC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CSb_Photo_PartialC(Z, shell, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_Photo_Partial


FUNCTION CS_Total_Kissel(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_Total_KisselC(Z, E, error) &
            BIND(C,NAME='CS_Total_Kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_Total_KisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_Total_KisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CS_Total_KisselC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_Total_Kissel


FUNCTION CSb_Total_Kissel(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_Total_KisselC(Z, E, error) &
            BIND(C,NAME='CSb_Total_Kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_Total_KisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_Total_KisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CSb_Total_KisselC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_Total_Kissel


FUNCTION ComptonProfile(Z, pz, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: pz
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION ComptonProfileC(Z, pz, error) &
            BIND(C,NAME='ComptonProfile')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: pz
                REAL (C_DOUBLE) :: ComptonProfileC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION ComptonProfileC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = ComptonProfileC(Z, pz, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION ComptonProfile


FUNCTION ComptonProfile_Partial(Z, shell, pz, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: shell
        REAL (C_DOUBLE), INTENT(IN) :: pz
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION ComptonProfile_PartialC(Z, shell, pz, error) &
            BIND(C,NAME='ComptonProfile_Partial')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: shell
                REAL (C_DOUBLE), INTENT(IN), VALUE :: pz
                REAL (C_DOUBLE) :: ComptonProfile_PartialC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION ComptonProfile_PartialC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = ComptonProfile_PartialC(Z, shell, pz, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION ComptonProfile_Partial


FUNCTION ElectronConfig(Z, shell, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: shell
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION ElectronConfigC(Z, shell, error) &
            BIND(C,NAME='ElectronConfig')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: shell
                REAL (C_DOUBLE) :: ElectronConfigC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION ElectronConfigC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = ElectronConfigC(Z, shell, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION ElectronConfig


FUNCTION ElectronConfig_Biggs(Z, shell, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: shell
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION ElectronConfig_BiggsC(Z, shell, error) &
            BIND(C,NAME='ElectronConfig_Biggs')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: shell
                REAL (C_DOUBLE) :: ElectronConfig_BiggsC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION ElectronConfig_BiggsC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = ElectronConfig_BiggsC(Z, shell, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION ElectronConfig_Biggs


FUNCTION AtomicLevelWidth(Z, shell, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: shell
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION AtomicLevelWidthC(Z, shell, error) &
            BIND(C,NAME='AtomicLevelWidth')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: shell
                REAL (C_DOUBLE) :: AtomicLevelWidthC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION AtomicLevelWidthC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = AtomicLevelWidthC(Z, shell, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION AtomicLevelWidth


FUNCTION AugerRate(Z, auger_trans, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: auger_trans
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION AugerRateC(Z, auger_trans, error) &
            BIND(C,NAME='AugerRate')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: auger_trans
                REAL (C_DOUBLE) :: AugerRateC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION AugerRateC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = AugerRateC(Z, auger_trans, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION AugerRate


FUNCTION AugerYield(Z, shell, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: shell
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION AugerYieldC(Z, shell, error) &
            BIND(C,NAME='AugerYield')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: shell
                REAL (C_DOUBLE) :: AugerYieldC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION AugerYieldC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = AugerYieldC(Z, shell, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION AugerYield


FUNCTION CS_FluorLine_Kissel(Z, line, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: line
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_FluorLine_KisselC(Z, line, E, error) &
            BIND(C,NAME='CS_FluorLine_Kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: line
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_FluorLine_KisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_FluorLine_KisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CS_FluorLine_KisselC(Z, line, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_FluorLine_Kissel


FUNCTION CSb_FluorLine_Kissel(Z, line, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: line
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_FluorLine_KisselC(Z, line, E, error) &
            BIND(C,NAME='CSb_FluorLine_Kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: line
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_FluorLine_KisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_FluorLine_KisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CSb_FluorLine_KisselC(Z, line, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_FluorLine_Kissel


FUNCTION CS_FluorLine_Kissel_Cascade(Z, line, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: line
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_FluorLine_Kissel_CascadeC(Z, line, E, error) &
            BIND(C,NAME='CS_FluorLine_Kissel_Cascade')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: line
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_FluorLine_Kissel_CascadeC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_FluorLine_Kissel_CascadeC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CS_FluorLine_Kissel_CascadeC(Z, line, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_FluorLine_Kissel_Cascade


FUNCTION CSb_FluorLine_Kissel_Cascade(Z, line, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: line
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_FluorLine_Kissel_CascadeC(Z, line, E, error) &
            BIND(C,NAME='CSb_FluorLine_Kissel_Cascade')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: line
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_FluorLine_Kissel_CascadeC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_FluorLine_Kissel_CascadeC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CSb_FluorLine_Kissel_CascadeC(Z, line, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_FluorLine_Kissel_Cascade


FUNCTION CS_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: line
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_FluorLine_Kissel_Nonradiative_CascadeC(Z, line, E, error) &
            BIND(C,NAME='CS_FluorLine_Kissel_Nonradiative_Cascade')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: line
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_FluorLine_Kissel_Nonradiative_CascadeC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_FluorLine_Kissel_Nonradiative_CascadeC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CS_FluorLine_Kissel_Nonradiative_CascadeC(Z, line, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_FluorLine_Kissel_Nonradiative_Cascade


FUNCTION CSb_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: line
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_FluorLine_Kissel_Nonradiative_CascadeC(Z, line, E, error) &
            BIND(C,NAME='CSb_FluorLine_Kissel_Nonradiative_Cascade')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: line
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_FluorLine_Kissel_Nonradiative_CascadeC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_FluorLine_Kissel_Nonradiative_CascadeC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CSb_FluorLine_Kissel_Nonradiative_CascadeC(Z, line, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_FluorLine_Kissel_Nonradiative_Cascade


FUNCTION CS_FluorLine_Kissel_Radiative_Cascade(Z, line, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: line
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_FluorLine_Kissel_Radiative_CascadeC(Z, line, E, error) &
            BIND(C,NAME='CS_FluorLine_Kissel_Radiative_Cascade')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: line
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_FluorLine_Kissel_Radiative_CascadeC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_FluorLine_Kissel_Radiative_CascadeC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CS_FluorLine_Kissel_Radiative_CascadeC(Z, line, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_FluorLine_Kissel_Radiative_Cascade


FUNCTION CSb_FluorLine_Kissel_Radiative_Cascade(Z, line, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: line
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_FluorLine_Kissel_Radiative_CascadeC(Z, line, E, error) &
            BIND(C,NAME='CSb_FluorLine_Kissel_Radiative_Cascade')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: line
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_FluorLine_Kissel_Radiative_CascadeC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_FluorLine_Kissel_Radiative_CascadeC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CSb_FluorLine_Kissel_Radiative_CascadeC(Z, line, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_FluorLine_Kissel_Radiative_Cascade


FUNCTION CS_FluorLine_Kissel_no_Cascade(Z, line, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: line
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_FluorLine_Kissel_no_CascadeC(Z, line, E, error) &
            BIND(C,NAME='CS_FluorLine_Kissel_no_Cascade')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: line
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_FluorLine_Kissel_no_CascadeC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_FluorLine_Kissel_no_CascadeC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CS_FluorLine_Kissel_no_CascadeC(Z, line, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_FluorLine_Kissel_no_Cascade


FUNCTION CSb_FluorLine_Kissel_no_Cascade(Z, line, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        INTEGER (C_INT), INTENT(IN) :: line
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_FluorLine_Kissel_no_CascadeC(Z, line, E, error) &
            BIND(C,NAME='CSb_FluorLine_Kissel_no_Cascade')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                INTEGER (C_INT), INTENT(IN), VALUE :: line
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_FluorLine_Kissel_no_CascadeC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_FluorLine_Kissel_no_CascadeC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = CSb_FluorLine_Kissel_no_CascadeC(Z, line, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_FluorLine_Kissel_no_Cascade


FUNCTION PL1_pure_kissel(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PL1_pure_kisselC(Z, E, error) &
            BIND(C,NAME='PL1_pure_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: PL1_pure_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PL1_pure_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PL1_pure_kisselC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PL1_pure_kissel


FUNCTION PL1_rad_cascade_kissel(Z, E, PK, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PL1_rad_cascade_kisselC(Z, E, PK, error) &
            BIND(C,NAME='PL1_rad_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE) :: PL1_rad_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PL1_rad_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PL1_rad_cascade_kisselC(Z, E, PK, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PL1_rad_cascade_kissel


FUNCTION PL1_auger_cascade_kissel(Z, E, PK, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PL1_auger_cascade_kisselC(Z, E, PK, error) &
            BIND(C,NAME='PL1_auger_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE) :: PL1_auger_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PL1_auger_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PL1_auger_cascade_kisselC(Z, E, PK, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PL1_auger_cascade_kissel


FUNCTION PL1_full_cascade_kissel(Z, E, PK, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PL1_full_cascade_kisselC(Z, E, PK, error) &
            BIND(C,NAME='PL1_full_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE) :: PL1_full_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PL1_full_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PL1_full_cascade_kisselC(Z, E, PK, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PL1_full_cascade_kissel


FUNCTION PL2_pure_kissel(Z, E, PL1, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PL2_pure_kisselC(Z, E, PL1, error) &
            BIND(C,NAME='PL2_pure_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE) :: PL2_pure_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PL2_pure_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PL2_pure_kisselC(Z, E, PL1, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PL2_pure_kissel


FUNCTION PL2_rad_cascade_kissel(Z, E, PK, PL1, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PL2_rad_cascade_kisselC(Z, E, PK, PL1, error) &
            BIND(C,NAME='PL2_rad_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE) :: PL2_rad_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PL2_rad_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PL2_rad_cascade_kisselC(Z, E, PK, PL1, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PL2_rad_cascade_kissel


FUNCTION PL2_auger_cascade_kissel(Z, E, PK, PL1, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PL2_auger_cascade_kisselC(Z, E, PK, PL1, error) &
            BIND(C,NAME='PL2_auger_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE) :: PL2_auger_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PL2_auger_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PL2_auger_cascade_kisselC(Z, E, PK, PL1, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PL2_auger_cascade_kissel


FUNCTION PL2_full_cascade_kissel(Z, E, PK, PL1, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PL2_full_cascade_kisselC(Z, E, PK, PL1, error) &
            BIND(C,NAME='PL2_full_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE) :: PL2_full_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PL2_full_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PL2_full_cascade_kisselC(Z, E, PK, PL1, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PL2_full_cascade_kissel


FUNCTION PL3_pure_kissel(Z, E, PL1, PL2, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PL3_pure_kisselC(Z, E, PL1, PL2, error) &
            BIND(C,NAME='PL3_pure_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE) :: PL3_pure_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PL3_pure_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PL3_pure_kisselC(Z, E, PL1, PL2, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PL3_pure_kissel


FUNCTION PL3_rad_cascade_kissel(Z, E, PK, PL1, PL2, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PL3_rad_cascade_kisselC(Z, E, PK, PL1, PL2, error) &
            BIND(C,NAME='PL3_rad_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE) :: PL3_rad_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PL3_rad_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PL3_rad_cascade_kisselC(Z, E, PK, PL1, PL2, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PL3_rad_cascade_kissel


FUNCTION PL3_auger_cascade_kissel(Z, E, PK, PL1, PL2, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PL3_auger_cascade_kisselC(Z, E, PK, PL1, PL2, error) &
            BIND(C,NAME='PL3_auger_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE) :: PL3_auger_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PL3_auger_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PL3_auger_cascade_kisselC(Z, E, PK, PL1, PL2, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PL3_auger_cascade_kissel


FUNCTION PL3_full_cascade_kissel(Z, E, PK, PL1, PL2, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PL3_full_cascade_kisselC(Z, E, PK, PL1, PL2, error) &
            BIND(C,NAME='PL3_full_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE) :: PL3_full_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PL3_full_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PL3_full_cascade_kisselC(Z, E, PK, PL1, PL2, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PL3_full_cascade_kissel


FUNCTION PM1_pure_kissel(Z, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM1_pure_kisselC(Z, E, error) &
            BIND(C,NAME='PM1_pure_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: PM1_pure_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM1_pure_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM1_pure_kisselC(Z, E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM1_pure_kissel


FUNCTION PM1_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE), INTENT(IN) :: PL3
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM1_rad_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, error) &
            BIND(C,NAME='PM1_rad_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL3
                REAL (C_DOUBLE) :: PM1_rad_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM1_rad_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM1_rad_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM1_rad_cascade_kissel


FUNCTION PM1_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE), INTENT(IN) :: PL3
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM1_auger_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, error) &
            BIND(C,NAME='PM1_auger_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL3
                REAL (C_DOUBLE) :: PM1_auger_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM1_auger_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM1_auger_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM1_auger_cascade_kissel


FUNCTION PM1_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE), INTENT(IN) :: PL3
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM1_full_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, error) &
            BIND(C,NAME='PM1_full_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL3
                REAL (C_DOUBLE) :: PM1_full_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM1_full_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM1_full_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM1_full_cascade_kissel


FUNCTION PM2_pure_kissel(Z, E, PM1, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PM1
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM2_pure_kisselC(Z, E, PM1, error) &
            BIND(C,NAME='PM2_pure_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM1
                REAL (C_DOUBLE) :: PM2_pure_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM2_pure_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM2_pure_kisselC(Z, E, PM1, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM2_pure_kissel


FUNCTION PM2_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE), INTENT(IN) :: PL3
        REAL (C_DOUBLE), INTENT(IN) :: PM1
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM2_rad_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, error) &
            BIND(C,NAME='PM2_rad_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL3
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM1
                REAL (C_DOUBLE) :: PM2_rad_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM2_rad_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM2_rad_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM2_rad_cascade_kissel


FUNCTION PM2_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE), INTENT(IN) :: PL3
        REAL (C_DOUBLE), INTENT(IN) :: PM1
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM2_auger_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, error) &
            BIND(C,NAME='PM2_auger_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL3
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM1
                REAL (C_DOUBLE) :: PM2_auger_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM2_auger_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM2_auger_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM2_auger_cascade_kissel


FUNCTION PM2_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE), INTENT(IN) :: PL3
        REAL (C_DOUBLE), INTENT(IN) :: PM1
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM2_full_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, error) &
            BIND(C,NAME='PM2_full_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL3
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM1
                REAL (C_DOUBLE) :: PM2_full_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM2_full_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM2_full_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM2_full_cascade_kissel


FUNCTION PM3_pure_kissel(Z, E, PM1, PM2, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PM1
        REAL (C_DOUBLE), INTENT(IN) :: PM2
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM3_pure_kisselC(Z, E, PM1, PM2, error) &
            BIND(C,NAME='PM3_pure_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM2
                REAL (C_DOUBLE) :: PM3_pure_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM3_pure_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM3_pure_kisselC(Z, E, PM1, PM2, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM3_pure_kissel


FUNCTION PM3_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE), INTENT(IN) :: PL3
        REAL (C_DOUBLE), INTENT(IN) :: PM1
        REAL (C_DOUBLE), INTENT(IN) :: PM2
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM3_rad_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, error) &
            BIND(C,NAME='PM3_rad_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL3
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM2
                REAL (C_DOUBLE) :: PM3_rad_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM3_rad_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM3_rad_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM3_rad_cascade_kissel


FUNCTION PM3_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE), INTENT(IN) :: PL3
        REAL (C_DOUBLE), INTENT(IN) :: PM1
        REAL (C_DOUBLE), INTENT(IN) :: PM2
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM3_auger_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, error) &
            BIND(C,NAME='PM3_auger_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL3
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM2
                REAL (C_DOUBLE) :: PM3_auger_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM3_auger_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM3_auger_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM3_auger_cascade_kissel


FUNCTION PM3_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE), INTENT(IN) :: PL3
        REAL (C_DOUBLE), INTENT(IN) :: PM1
        REAL (C_DOUBLE), INTENT(IN) :: PM2
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM3_full_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, error) &
            BIND(C,NAME='PM3_full_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL3
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM2
                REAL (C_DOUBLE) :: PM3_full_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM3_full_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM3_full_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM3_full_cascade_kissel


FUNCTION PM4_pure_kissel(Z, E, PM1, PM2, PM3, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PM1
        REAL (C_DOUBLE), INTENT(IN) :: PM2
        REAL (C_DOUBLE), INTENT(IN) :: PM3
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM4_pure_kisselC(Z, E, PM1, PM2, PM3, error) &
            BIND(C,NAME='PM4_pure_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM3
                REAL (C_DOUBLE) :: PM4_pure_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM4_pure_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM4_pure_kisselC(Z, E, PM1, PM2, PM3, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM4_pure_kissel


FUNCTION PM4_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE), INTENT(IN) :: PL3
        REAL (C_DOUBLE), INTENT(IN) :: PM1
        REAL (C_DOUBLE), INTENT(IN) :: PM2
        REAL (C_DOUBLE), INTENT(IN) :: PM3
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM4_rad_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, error) &
            BIND(C,NAME='PM4_rad_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL3
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM3
                REAL (C_DOUBLE) :: PM4_rad_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM4_rad_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM4_rad_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM4_rad_cascade_kissel


FUNCTION PM4_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE), INTENT(IN) :: PL3
        REAL (C_DOUBLE), INTENT(IN) :: PM1
        REAL (C_DOUBLE), INTENT(IN) :: PM2
        REAL (C_DOUBLE), INTENT(IN) :: PM3
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM4_auger_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, error) &
            BIND(C,NAME='PM4_auger_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL3
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM3
                REAL (C_DOUBLE) :: PM4_auger_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM4_auger_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM4_auger_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM4_auger_cascade_kissel


FUNCTION PM4_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE), INTENT(IN) :: PL3
        REAL (C_DOUBLE), INTENT(IN) :: PM1
        REAL (C_DOUBLE), INTENT(IN) :: PM2
        REAL (C_DOUBLE), INTENT(IN) :: PM3
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM4_full_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, error) &
            BIND(C,NAME='PM4_full_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL3
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM3
                REAL (C_DOUBLE) :: PM4_full_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM4_full_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM4_full_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM4_full_cascade_kissel


FUNCTION PM5_pure_kissel(Z, E, PM1, PM2, PM3, PM4, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PM1
        REAL (C_DOUBLE), INTENT(IN) :: PM2
        REAL (C_DOUBLE), INTENT(IN) :: PM3
        REAL (C_DOUBLE), INTENT(IN) :: PM4
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM5_pure_kisselC(Z, E, PM1, PM2, PM3, PM4, error) &
            BIND(C,NAME='PM5_pure_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM3
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM4
                REAL (C_DOUBLE) :: PM5_pure_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM5_pure_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM5_pure_kisselC(Z, E, PM1, PM2, PM3, PM4, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM5_pure_kissel


FUNCTION PM5_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE), INTENT(IN) :: PL3
        REAL (C_DOUBLE), INTENT(IN) :: PM1
        REAL (C_DOUBLE), INTENT(IN) :: PM2
        REAL (C_DOUBLE), INTENT(IN) :: PM3
        REAL (C_DOUBLE), INTENT(IN) :: PM4
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM5_rad_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, error) &
            BIND(C,NAME='PM5_rad_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL3
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM3
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM4
                REAL (C_DOUBLE) :: PM5_rad_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM5_rad_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM5_rad_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM5_rad_cascade_kissel


FUNCTION PM5_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE), INTENT(IN) :: PL3
        REAL (C_DOUBLE), INTENT(IN) :: PM1
        REAL (C_DOUBLE), INTENT(IN) :: PM2
        REAL (C_DOUBLE), INTENT(IN) :: PM3
        REAL (C_DOUBLE), INTENT(IN) :: PM4
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM5_auger_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, error) &
            BIND(C,NAME='PM5_auger_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL3
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM3
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM4
                REAL (C_DOUBLE) :: PM5_auger_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM5_auger_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM5_auger_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM5_auger_cascade_kissel


FUNCTION PM5_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        INTEGER (C_INT), INTENT(IN) :: Z
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: PK
        REAL (C_DOUBLE), INTENT(IN) :: PL1
        REAL (C_DOUBLE), INTENT(IN) :: PL2
        REAL (C_DOUBLE), INTENT(IN) :: PL3
        REAL (C_DOUBLE), INTENT(IN) :: PM1
        REAL (C_DOUBLE), INTENT(IN) :: PM2
        REAL (C_DOUBLE), INTENT(IN) :: PM3
        REAL (C_DOUBLE), INTENT(IN) :: PM4
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION PM5_full_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, error) &
            BIND(C,NAME='PM5_full_cascade_kissel')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: Z
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PK
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PL3
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM1
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM2
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM3
                REAL (C_DOUBLE), INTENT(IN), VALUE :: PM4
                REAL (C_DOUBLE) :: PM5_full_cascade_kisselC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION PM5_full_cascade_kisselC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        
        rv = PM5_full_cascade_kisselC(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION PM5_full_cascade_kissel


FUNCTION CS_Total_CP(compound, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_Total_CPC(compound, E, error) &
            BIND(C,NAME='CS_Total_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_Total_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_Total_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = CS_Total_CPC(C_LOC(compound_F), E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_Total_CP


FUNCTION CS_Photo_CP(compound, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_Photo_CPC(compound, E, error) &
            BIND(C,NAME='CS_Photo_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_Photo_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_Photo_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = CS_Photo_CPC(C_LOC(compound_F), E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_Photo_CP


FUNCTION CS_Rayl_CP(compound, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_Rayl_CPC(compound, E, error) &
            BIND(C,NAME='CS_Rayl_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_Rayl_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_Rayl_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = CS_Rayl_CPC(C_LOC(compound_F), E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_Rayl_CP


FUNCTION CS_Compt_CP(compound, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_Compt_CPC(compound, E, error) &
            BIND(C,NAME='CS_Compt_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_Compt_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_Compt_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = CS_Compt_CPC(C_LOC(compound_F), E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_Compt_CP


FUNCTION CS_Energy_CP(compound, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_Energy_CPC(compound, E, error) &
            BIND(C,NAME='CS_Energy_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_Energy_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_Energy_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = CS_Energy_CPC(C_LOC(compound_F), E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_Energy_CP


FUNCTION CSb_Total_CP(compound, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_Total_CPC(compound, E, error) &
            BIND(C,NAME='CSb_Total_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_Total_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_Total_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = CSb_Total_CPC(C_LOC(compound_F), E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_Total_CP


FUNCTION CSb_Photo_CP(compound, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_Photo_CPC(compound, E, error) &
            BIND(C,NAME='CSb_Photo_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_Photo_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_Photo_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = CSb_Photo_CPC(C_LOC(compound_F), E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_Photo_CP


FUNCTION CSb_Rayl_CP(compound, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_Rayl_CPC(compound, E, error) &
            BIND(C,NAME='CSb_Rayl_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_Rayl_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_Rayl_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = CSb_Rayl_CPC(C_LOC(compound_F), E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_Rayl_CP


FUNCTION CSb_Compt_CP(compound, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_Compt_CPC(compound, E, error) &
            BIND(C,NAME='CSb_Compt_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_Compt_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_Compt_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = CSb_Compt_CPC(C_LOC(compound_F), E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_Compt_CP


FUNCTION DCS_Rayl_CP(compound, E, theta, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCS_Rayl_CPC(compound, E, theta, error) &
            BIND(C,NAME='DCS_Rayl_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE) :: DCS_Rayl_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCS_Rayl_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = DCS_Rayl_CPC(C_LOC(compound_F), E, theta, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCS_Rayl_CP


FUNCTION DCS_Compt_CP(compound, E, theta, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCS_Compt_CPC(compound, E, theta, error) &
            BIND(C,NAME='DCS_Compt_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE) :: DCS_Compt_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCS_Compt_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = DCS_Compt_CPC(C_LOC(compound_F), E, theta, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCS_Compt_CP


FUNCTION DCSb_Rayl_CP(compound, E, theta, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCSb_Rayl_CPC(compound, E, theta, error) &
            BIND(C,NAME='DCSb_Rayl_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE) :: DCSb_Rayl_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCSb_Rayl_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = DCSb_Rayl_CPC(C_LOC(compound_F), E, theta, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCSb_Rayl_CP


FUNCTION DCSb_Compt_CP(compound, E, theta, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCSb_Compt_CPC(compound, E, theta, error) &
            BIND(C,NAME='DCSb_Compt_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE) :: DCSb_Compt_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCSb_Compt_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = DCSb_Compt_CPC(C_LOC(compound_F), E, theta, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCSb_Compt_CP


FUNCTION DCSP_Rayl_CP(compound, E, theta, phi, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE), INTENT(IN) :: phi
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCSP_Rayl_CPC(compound, E, theta, phi, error) &
            BIND(C,NAME='DCSP_Rayl_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE), INTENT(IN), VALUE :: phi
                REAL (C_DOUBLE) :: DCSP_Rayl_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCSP_Rayl_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = DCSP_Rayl_CPC(C_LOC(compound_F), E, theta, phi, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCSP_Rayl_CP


FUNCTION DCSP_Compt_CP(compound, E, theta, phi, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE), INTENT(IN) :: phi
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCSP_Compt_CPC(compound, E, theta, phi, error) &
            BIND(C,NAME='DCSP_Compt_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE), INTENT(IN), VALUE :: phi
                REAL (C_DOUBLE) :: DCSP_Compt_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCSP_Compt_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = DCSP_Compt_CPC(C_LOC(compound_F), E, theta, phi, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCSP_Compt_CP


FUNCTION DCSPb_Rayl_CP(compound, E, theta, phi, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE), INTENT(IN) :: phi
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCSPb_Rayl_CPC(compound, E, theta, phi, error) &
            BIND(C,NAME='DCSPb_Rayl_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE), INTENT(IN), VALUE :: phi
                REAL (C_DOUBLE) :: DCSPb_Rayl_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCSPb_Rayl_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = DCSPb_Rayl_CPC(C_LOC(compound_F), E, theta, phi, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCSPb_Rayl_CP


FUNCTION DCSPb_Compt_CP(compound, E, theta, phi, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: theta
        REAL (C_DOUBLE), INTENT(IN) :: phi
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION DCSPb_Compt_CPC(compound, E, theta, phi, error) &
            BIND(C,NAME='DCSPb_Compt_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: theta
                REAL (C_DOUBLE), INTENT(IN), VALUE :: phi
                REAL (C_DOUBLE) :: DCSPb_Compt_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION DCSPb_Compt_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = DCSPb_Compt_CPC(C_LOC(compound_F), E, theta, phi, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION DCSPb_Compt_CP


FUNCTION CS_Photo_Total_CP(compound, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_Photo_Total_CPC(compound, E, error) &
            BIND(C,NAME='CS_Photo_Total_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_Photo_Total_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_Photo_Total_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = CS_Photo_Total_CPC(C_LOC(compound_F), E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_Photo_Total_CP


FUNCTION CS_Total_Kissel_CP(compound, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CS_Total_Kissel_CPC(compound, E, error) &
            BIND(C,NAME='CS_Total_Kissel_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CS_Total_Kissel_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CS_Total_Kissel_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = CS_Total_Kissel_CPC(C_LOC(compound_F), E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CS_Total_Kissel_CP


FUNCTION CSb_Photo_Total_CP(compound, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_Photo_Total_CPC(compound, E, error) &
            BIND(C,NAME='CSb_Photo_Total_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_Photo_Total_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_Photo_Total_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = CSb_Photo_Total_CPC(C_LOC(compound_F), E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_Photo_Total_CP


FUNCTION CSb_Total_Kissel_CP(compound, E, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION CSb_Total_Kissel_CPC(compound, E, error) &
            BIND(C,NAME='CSb_Total_Kissel_CP')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE) :: CSb_Total_Kissel_CPC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION CSb_Total_Kissel_CPC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = CSb_Total_Kissel_CPC(C_LOC(compound_F), E, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION CSb_Total_Kissel_CP


FUNCTION Refractive_Index_Re(compound, E, density, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: density
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION Refractive_Index_ReC(compound, E, density, error) &
            BIND(C,NAME='Refractive_Index_Re')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: density
                REAL (C_DOUBLE) :: Refractive_Index_ReC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION Refractive_Index_ReC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = Refractive_Index_ReC(C_LOC(compound_F), E, density, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION Refractive_Index_Re


FUNCTION Refractive_Index_Im(compound, E, density, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: compound 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        compound_F
        REAL (C_DOUBLE), INTENT(IN) :: E
        REAL (C_DOUBLE), INTENT(IN) :: density
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION Refractive_Index_ImC(compound, E, density, error) &
            BIND(C,NAME='Refractive_Index_Im')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN), VALUE :: compound
                REAL (C_DOUBLE), INTENT(IN), VALUE :: E
                REAL (C_DOUBLE), INTENT(IN), VALUE :: density
                REAL (C_DOUBLE) :: Refractive_Index_ImC
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION Refractive_Index_ImC
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        CALL stringF2C(compound, compound_F)

        rv = Refractive_Index_ImC(C_LOC(compound_F), E, density, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION Refractive_Index_Im
