PROGRAM test_radionuclides

USE, INTRINSIC :: ISO_C_BINDING
USE, INTRINSIC :: ISO_FORTRAN_ENV
USE :: xraylib
USE :: libtest
IMPLICIT NONE

TYPE(xrl_error), POINTER :: error => NULL()
INTEGER, PARAMETER :: RADIONUCLIDE_LIST_LEN = 10
CHARACTER(KIND=C_CHAR, LEN=RADIO_NUCLIDE_STRING_LENGTH), DIMENSION(:), POINTER :: radionuclide_list
INTEGER :: i
TYPE (radioNuclideData), POINTER :: c

radionuclide_list => GetRadioNuclideDataList(error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(SIZE(radionuclide_list) == RADIONUCLIDE_LIST_LEN)

DO i=1,RADIONUCLIDE_LIST_LEN
        c => GetRadioNuclideDataByName(radionuclide_list(i))
        CALL assert(c%name == radionuclide_list(i))
        DEALLOCATE(c)
ENDDO

DO i=1,RADIONUCLIDE_LIST_LEN
        c => GetRadioNuclideDataByIndex(i-1)
        CALL assert(c%name == radionuclide_list(i))
        DEALLOCATE(c)
ENDDO

DEALLOCATE(radionuclide_list)

c => GetRadioNuclideDataByIndex(3)
CALL assert(c%A == 125)
CALL assert(ALL(ABS(c%GammaEnergies - [35.4919_C_DOUBLE]) < 1E-6_C_DOUBLE))
CALL assert(ALL(ABS(c%GammaIntensities - [0.0668_C_DOUBLE]) < 1E-6_C_DOUBLE))
CALL assert(c%N == 72)
CALL assert(ALL(ABS(c%XrayIntensities- [0.0023_C_DOUBLE, 0.00112_C_DOUBLE, &
   0.0063_C_DOUBLE, 0.056_C_DOUBLE, 0.035_C_DOUBLE, 0.0042_C_DOUBLE, &
   0.007_C_DOUBLE, 0.00043_C_DOUBLE, 0.0101_C_DOUBLE, 0.0045_C_DOUBLE, &
   0.00103_C_DOUBLE, 0.0016_C_DOUBLE, 3.24e-05_C_DOUBLE, 0.406_C_DOUBLE, &
   0.757_C_DOUBLE, 0.0683_C_DOUBLE, 0.132_C_DOUBLE, 0.00121_C_DOUBLE, &
   0.0381_C_DOUBLE, 0.0058_C_DOUBLE]) < 1E-6_C_DOUBLE))
CALL assert(ALL(c%XrayLines - [-86, -60, -89, -90, -63, -33, -34, -91, -95,&
   -68, -38, -39, -1, -2, -3, -5, -6, -8, -11, -13] == 0))
CALL assert(c%Z == 53)
CALL assert(c%Z_xray == 52)
CALL assert(c%nGammas == 1)
CALL assert(c%nXrays == 20)
CALL assert(c%name == '125I')
DEALLOCATE(c)

c => GetRadioNuclideDataByName('125I')
CALL assert(c%A == 125)
CALL assert(ALL(ABS(c%GammaEnergies - [35.4919_C_DOUBLE]) < 1E-6_C_DOUBLE))
CALL assert(ALL(ABS(c%GammaIntensities - [0.0668_C_DOUBLE]) < 1E-6_C_DOUBLE))
CALL assert(c%N == 72)
CALL assert(ALL(ABS(c%XrayIntensities- [0.0023_C_DOUBLE, 0.00112_C_DOUBLE, &
   0.0063_C_DOUBLE, 0.056_C_DOUBLE, 0.035_C_DOUBLE, 0.0042_C_DOUBLE, &
   0.007_C_DOUBLE, 0.00043_C_DOUBLE, 0.0101_C_DOUBLE, 0.0045_C_DOUBLE, &
   0.00103_C_DOUBLE, 0.0016_C_DOUBLE, 3.24e-05_C_DOUBLE, 0.406_C_DOUBLE, &
   0.757_C_DOUBLE, 0.0683_C_DOUBLE, 0.132_C_DOUBLE, 0.00121_C_DOUBLE, &
   0.0381_C_DOUBLE, 0.0058_C_DOUBLE]) < 1E-6_C_DOUBLE))
CALL assert(ALL(c%XrayLines - [-86, -60, -89, -90, -63, -33, -34, -91, -95,&
   -68, -38, -39, -1, -2, -3, -5, -6, -8, -11, -13] == 0))
CALL assert(c%Z == 53)
CALL assert(c%Z_xray == 52)
CALL assert(c%nGammas == 1)
CALL assert(c%nXrays == 20)
CALL assert(c%name == '125I')
DEALLOCATE(c)

c => GetRadioNuclideDataByIndex(-1)
CALL assert(.NOT. ASSOCIATED(c))
c => GetRadioNuclideDataByIndex(RADIONUCLIDE_LIST_LEN)
CALL assert(.NOT. ASSOCIATED(c))
c => GetRadioNuclideDataByName('non-existent-radionuclide')
CALL assert(.NOT. ASSOCIATED(c))

c => GetRadioNuclideDataByIndex(-1, error)
CALL assert(.NOT. ASSOCIATED(c))
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)
c => GetRadioNuclideDataByIndex(RADIONUCLIDE_LIST_LEN, error)
CALL assert(.NOT. ASSOCIATED(c))
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)
c => GetRadioNuclideDataByName('non-existent-radionuclide', error)
CALL assert(.NOT. ASSOCIATED(c))
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

ENDPROGRAM test_radionuclides

