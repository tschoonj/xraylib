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
