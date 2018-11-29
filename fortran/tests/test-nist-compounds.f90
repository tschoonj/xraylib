PROGRAM test_nist_compounds

USE, INTRINSIC :: ISO_C_BINDING
USE, INTRINSIC :: ISO_FORTRAN_ENV
USE :: xraylib
USE :: libtest
IMPLICIT NONE

TYPE(xrl_error), POINTER :: error => NULL()
INTEGER, PARAMETER :: NIST_LIST_LEN = 180
CHARACTER(KIND=C_CHAR, LEN=NIST_LIST_STRING_LENGTH), DIMENSION(:), POINTER :: nist_list
INTEGER :: i
TYPE (compoundDataNIST), POINTER :: c

nist_list => GetCompoundDataNISTList(error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(SIZE(nist_list) == NIST_LIST_LEN)

DO i=1,NIST_LIST_LEN
        c => GetCompoundDataNISTByName(nist_list(i))
        CALL assert(c%name == nist_list(i))
        DEALLOCATE(c)
ENDDO

DO i=1,NIST_LIST_LEN
        c => GetCompoundDataNISTByIndex(i-1)
        CALL assert(c%name == nist_list(i))
        DEALLOCATE(c)
ENDDO

DEALLOCATE(nist_list)

c => GetCompoundDataNISTByIndex(5)
CALL assert(c%nElements == 4)
CALL assert(ABS(c%density - 0.001205) < 1E-6_C_DOUBLE)
CALL assert(ALL(c%Elements - [6, 7, 8, 18] == 0))
CALL assert(ALL(ABS(c%massFractions- [0.000124, 0.755267, 0.231781, 0.012827]) < 1E-6))
CALL assert(c%name == 'Air, Dry (near sea level)')
DEALLOCATE(c)

c => GetCompoundDataNISTByName('Air, Dry (near sea level)')
CALL assert(c%nElements == 4)
CALL assert(ABS(c%density - 0.001205) < 1E-6_C_DOUBLE)
CALL assert(ALL(c%Elements - [6, 7, 8, 18] == 0))
CALL assert(ALL(ABS(c%massFractions- [0.000124, 0.755267, 0.231781, 0.012827]) < 1E-6))
CALL assert(c%name == 'Air, Dry (near sea level)')
DEALLOCATE(c)

c => GetCompoundDataNISTByIndex(-1)
CALL assert(.NOT. ASSOCIATED(c))
c => GetCompoundDataNISTByIndex(NIST_LIST_LEN)
CALL assert(.NOT. ASSOCIATED(c))
c => GetCompoundDataNISTByName('non-existent-compound')
CALL assert(.NOT. ASSOCIATED(c))

c => GetCompoundDataNISTByIndex(-1, error)
CALL assert(.NOT. ASSOCIATED(c))
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)
c => GetCompoundDataNISTByIndex(NIST_LIST_LEN, error)
CALL assert(.NOT. ASSOCIATED(c))
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)
c => GetCompoundDataNISTByName('non-existent-compound', error)
CALL assert(.NOT. ASSOCIATED(c))
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

ENDPROGRAM test_nist_compounds
