PROGRAM test_crystal_diffraction

USE, INTRINSIC :: ISO_C_BINDING
USE, INTRINSIC :: ISO_FORTRAN_ENV
USE :: xraylib
USE :: libtest
IMPLICIT NONE

TYPE (xrl_error), POINTER :: error => NULL()
INTEGER, PARAMETER :: CRYSTALS_LIST_LEN = 38
CHARACTER(KIND=C_CHAR, LEN=CRYSTAL_STRING_LENGTH), DIMENSION(:), POINTER :: crystal_list
INTEGER :: i
TYPE (Crystal_Struct), POINTER :: cs, cs2
TYPE (Crystal_Atom) :: ca, ca2
INTEGER (C_INT) :: current_ncrystals, rv
REAL (C_DOUBLE) :: tmp, f0, f_prime, f_prime2
REAL (C_DOUBLE), PARAMETER :: PI = 4 * ATAN(1.0_C_DOUBLE)

crystal_list => Crystal_GetCrystalsList(error=error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(SIZE(crystal_list) == CRYSTALS_LIST_LEN)

DO i=1,CRYSTALS_LIST_LEN
        cs => Crystal_GetCrystal(crystal_list(i), error=error)
        CALL assert(.NOT. ASSOCIATED(error))
        CALL assert(cs%name == crystal_list(i))
        DEALLOCATE(cs)
ENDDO

DEALLOCATE(crystal_list)

cs => Crystal_GetCrystal('')
CALL assert(.NOT. ASSOCIATED(cs))

cs => Crystal_GetCrystal('', error=error)
CALL assert(.NOT. ASSOCIATED(cs))
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

cs => Crystal_GetCrystal('non-existent-crystal', error=error)
CALL assert(.NOT. ASSOCIATED(cs))
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

cs => Crystal_GetCrystal('')
CALL assert(.NOT. ASSOCIATED(cs))

cs => Crystal_GetCrystal('non-existent-crystal')
CALL assert(.NOT. ASSOCIATED(cs))

cs => Crystal_GetCrystal('Diamond', error=error)
CALL assert(ASSOCIATED(cs))
CALL assert(.NOT. ASSOCIATED(error))

cs2 => Crystal_MakeCopy(cs)
CALL assert(ASSOCIATED(cs2))
ca = cs%atom(1)
ca2 = cs2%atom(1)
DEALLOCATE(cs)
CALL assert(cs2%name == 'Diamond')
CALL assert(ca%Zatom == ca2%Zatom)
CALL assert(ABS(ca%fraction - ca2%fraction) .LT. 1E-6_C_DOUBLE)
CALL assert(ABS(ca%x - ca2%x) .LT. 1E-6_C_DOUBLE)
CALL assert(ABS(ca%y - ca2%y) .LT. 1E-6_C_DOUBLE)
CALL assert(ABS(ca%z - ca2%z) .LT. 1E-6_C_DOUBLE)
DEALLOCATE(cs2)

cs => Crystal_GetCrystal('Diamond', error=error)
CALL assert(ASSOCIATED(cs))
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(Crystal_AddCrystal(cs, error=error) == 0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

cs2 => Crystal_MakeCopy(cs)
CALL assert(ASSOCIATED(cs2))
cs2%name = 'Diamond-copy'
CALL assert(Crystal_AddCrystal(cs2, error=error) == 1)

CALL assert(Crystal_AddCrystal(cs, error=error) == 0)
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

crystal_list => Crystal_GetCrystalsList(error=error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(SIZE(crystal_list) == CRYSTALS_LIST_LEN + 1)
DEALLOCATE(crystal_list)

current_ncrystals = CRYSTALS_LIST_LEN + 1

DO i=1, CRYSTALARRAY_MAX
        cs2 => Crystal_MakeCopy(cs)
        CALL assert(ASSOCIATED(cs2))
        WRITE (cs2%name, '(A, I0)') 'Diamond copy ', i
        rv = Crystal_AddCrystal(cs2, error=error)
        IF (current_ncrystals .LT. CRYSTALARRAY_MAX) THEN
                CALL assert(.NOT. ASSOCIATED(error))
                CALL assert(rv == 1)
                crystal_list => Crystal_GetCrystalsList(error=error)
                CALL assert(.NOT. ASSOCIATED(error))
                current_ncrystals = current_ncrystals + 1
                CALL assert(SIZE(crystal_list) == current_ncrystals)
                DEALLOCATE(crystal_list)
        ELSE
                CALL assert(ASSOCIATED(error))
                CALL assert(rv == 0)
                CALL assert(error%code == XRL_ERROR_RUNTIME)
                DEALLOCATE(error)
                crystal_list => Crystal_GetCrystalsList(error=error)
                CALL assert(.NOT. ASSOCIATED(error))
                CALL assert(SIZE(crystal_list) == CRYSTALARRAY_MAX)
                DEALLOCATE(crystal_list)
        ENDIF
        DEALLOCATE(cs2)
ENDDO

! bragg angle
tmp = Bragg_angle(cs, 10.0_C_DOUBLE, 1, 1, 1, error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(ABS(tmp - 0.3057795845795849) < 1E-6)

tmp = Bragg_angle(cs, -10.0_C_DOUBLE, 1, 1, 1, error)
CALL assert(tmp == 0.0);
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

tmp = Bragg_angle(cs, 10.0_C_DOUBLE, 0, 0, 0, error)
CALL assert(tmp == 0.0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

! Q_scattering_amplitude
tmp = Q_scattering_amplitude(cs, 10.0_C_DOUBLE, 1, 1, 1, PI/4.0, error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(ABS(tmp - 0.19184445408324474) < 1E-6)

tmp = Q_scattering_amplitude(cs, -10.0_C_DOUBLE, 1, 1, 1, PI/4, error)
CALL assert(tmp == 0.0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

tmp = Q_scattering_amplitude(cs, 10.0_C_DOUBLE, 0, 0, 0, PI/4, error)
CALL assert(tmp == 0.0)
CALL assert(.NOT. ASSOCIATED(error))
! Atomic_Factors
rv = Atomic_Factors(26, 10.0_C_DOUBLE, 1.0_C_DOUBLE, 10.0_C_DOUBLE, f0, f_prime, f_prime2, error)
CALL assert(rv == 1)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(ABS(f0 - 65.15) < 1E-5)
CALL assert(ABS(f_prime + 0.22193271025027966) < 1E-6)
CALL assert(ABS(f_prime2 - 22.420270655080493) < 1E-6)

rv = Atomic_Factors(-1, 10.0_C_DOUBLE, 1.0_C_DOUBLE, 10.0_C_DOUBLE, f0, f_prime, f_prime2, error)
CALL assert(rv == 0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
CALL assert(f0 == 0.0)
CALL assert(f_prime == 0.0)
CALL assert(f_prime2 == 0.0)
DEALLOCATE(error)

rv = Atomic_Factors(26, -10.0_C_DOUBLE, 1.0_C_DOUBLE, 10.0_C_DOUBLE, f0, f_prime, f_prime2, error)
CALL assert(rv == 0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
CALL assert(f0 == 0.0)
CALL assert(f_prime == 0.0)
CALL assert(f_prime2 == 0.0)
DEALLOCATE(error)

rv = Atomic_Factors(26, 10.0_C_DOUBLE, -1.0_C_DOUBLE, 10.0_C_DOUBLE, f0, f_prime, f_prime2, error)
CALL assert(rv == 0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
CALL assert(f0 == 0.0)
CALL assert(f_prime == 0.0)
CALL assert(f_prime2 == 0.0)
DEALLOCATE(error)

rv = Atomic_Factors(26, 10.0_C_DOUBLE, 1.0_C_DOUBLE, -10.0_C_DOUBLE, f0, f_prime, f_prime2, error)
CALL assert(rv == 0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
CALL assert(f0 == 0.0)
CALL assert(f_prime == 0.0)
CALL assert(f_prime2 == 0.0)
DEALLOCATE(error)

!	rv = Atomic_Factors(26, 10, 1.0, 10.0, NULL, &f_prime, &f_prime2, &error);
!	assert(rv == 1);
!	assert(error == NULL);
!	assert(f0 == 0.0);
!	assert(f_prime != 0.0);
!	assert(f_prime2 != 0.0);
!		
!	rv = Atomic_Factors(26, 10, 1.0, 10.0, &f0, NULL, &f_prime2, &error);
!	assert(rv == 1);
!	assert(error == NULL);
!	assert(f0 != 0.0);
!	assert(f_prime2 != 0.0);
!		
!	rv = Atomic_Factors(26, 10, 1.0, 10.0, &f0, &f_prime, NULL, &error);
!	assert(rv == 1);
!	assert(error == NULL);
!	assert(f0 != 0.0);
!	assert(f_prime != 0.0);

! unit cell volume
tmp = Crystal_UnitCellVolume(cs, error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(ABS(tmp - 45.376673902751) < 1E-6)

! crystal dspacing
tmp = Crystal_dSpacing(cs, 1, 1, 1, error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(ABS(tmp - 2.0592870875248344) < 1E-6)

tmp = Crystal_dSpacing(cs, 0, 0, 0, error)
CALL assert(ASSOCIATED(error))
CALL assert(tmp == 0.0)
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

DEALLOCATE(cs)
ENDPROGRAM test_crystal_diffraction
