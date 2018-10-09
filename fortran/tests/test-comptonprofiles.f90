PROGRAM test_comptonprofiles

USE, INTRINSIC :: ISO_C_BINDING
USE, INTRINSIC :: ISO_FORTRAN_ENV
USE :: xraylib
USE :: libtest
IMPLICIT NONE

TYPE(xrl_error), POINTER :: error => NULL()
REAL (C_DOUBLE) :: profile, profile1, profile2

! pz == 0.0
profile = ComptonProfile(26, 0.0_C_DOUBLE, error)
CALL assert(abs(profile - 7.060_C_DOUBLE) < 1E-6)
CALL assert(.NOT. ASSOCIATED(error))

profile = ComptonProfile_Partial(26, N1_SHELL, 0.0_C_DOUBLE, error)
CALL assert(abs(profile - 1.550_C_DOUBLE) < 1E-6)
CALL assert(.NOT. ASSOCIATED(error))

profile1 = ComptonProfile_Partial(26, L2_SHELL, 0.0_C_DOUBLE, error)
profile2 = ComptonProfile_Partial(26, L3_SHELL, 0.0_C_DOUBLE, error)
CALL assert(abs(profile1 - profile2) < 1E-6)
CALL assert(abs(profile1 - 0.065_C_DOUBLE) < 1E-6)
CALL assert(.NOT. ASSOCIATED(error))

! pz == 100.0
profile = ComptonProfile(26, 100.0_C_DOUBLE, error)
CALL assert(abs(profile - 1.800E-05_C_DOUBLE) < 1E-8)
CALL assert(.NOT. ASSOCIATED(error))

profile = ComptonProfile_Partial(26, N1_SHELL, 100.0_C_DOUBLE, error)
CALL assert(abs(profile - 5.100E-09_C_DOUBLE) < 1E-12)
CALL assert(.NOT. ASSOCIATED(error))

profile1 = ComptonProfile_Partial(26, L2_SHELL, 100.0_C_DOUBLE, error)
profile2 = ComptonProfile_Partial(26, L3_SHELL, 100.0_C_DOUBLE, error)
CALL assert(abs(profile1 - profile2) < 1E-10)
CALL assert(abs(profile1 - 1.100E-08_C_DOUBLE) < 1E-10)
CALL assert(.NOT. ASSOCIATED(error))

! pz == 50.0 -> interpolated!
profile = ComptonProfile(26, 50.0_C_DOUBLE, error)
CALL assert(abs(profile - 0.0006843950273082384_C_DOUBLE) < 1E-8)
CALL assert(.NOT. ASSOCIATED(error))

profile = ComptonProfile_Partial(26, N1_SHELL, 50.0_C_DOUBLE, error)
CALL assert(abs(profile - 2.4322755767709126e-07_C_DOUBLE) < 1E-10)
CALL assert(.NOT. ASSOCIATED(error))

profile1 = ComptonProfile_Partial(26, L2_SHELL, 50.0_C_DOUBLE, error)
profile2 = ComptonProfile_Partial(26, L3_SHELL, 50.0_C_DOUBLE, error)
CALL assert(abs(profile1 - profile2) < 1E-10)
CALL assert(abs(profile1 - 2.026953933016568e-06_C_DOUBLE) < 1E-10)
CALL assert(.NOT. ASSOCIATED(error))

! bad input
profile = ComptonProfile(0, 0.0_C_DOUBLE)
CALL assert(profile == 0.0)
profile = ComptonProfile(0, 0.0_C_DOUBLE, error)
CALL assert(profile == 0.0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

profile = ComptonProfile(102, 0.0_C_DOUBLE, error)
CALL assert(.NOT. ASSOCIATED(error))

profile = ComptonProfile(103, 0.0_C_DOUBLE)
CALL assert(profile == 0.0)
profile = ComptonProfile(103, 0.0_C_DOUBLE, error)
CALL assert(profile == 0.0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

profile = ComptonProfile(26, -1.0_C_DOUBLE)
CALL assert(profile == 0.0)
profile = ComptonProfile(26, -1.0_C_DOUBLE, error)
CALL assert(profile == 0.0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

profile = ComptonProfile_Partial(0, K_SHELL, 0.0_C_DOUBLE)
CALL assert(profile == 0.0)
profile = ComptonProfile_Partial(0, K_SHELL, 0.0_C_DOUBLE, error)
CALL assert(profile == 0.0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

profile = ComptonProfile_Partial(102, K_SHELL, 0.0_C_DOUBLE, error)
CALL assert(.NOT. ASSOCIATED(error))

profile = ComptonProfile_Partial(103, K_SHELL, 0.0_C_DOUBLE)
CALL assert(profile == 0.0)
profile = ComptonProfile_Partial(103, K_SHELL, 0.0_C_DOUBLE, error)
CALL assert(profile == 0.0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

profile = ComptonProfile_Partial(26, K_SHELL, -1.0_C_DOUBLE)
CALL assert(profile == 0.0)
profile = ComptonProfile_Partial(26, K_SHELL, -1.0_C_DOUBLE, error)
CALL assert(profile == 0.0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

profile = ComptonProfile_Partial(26, -1, 0.0_C_DOUBLE)
CALL assert(profile == 0.0)
profile = ComptonProfile_Partial(26, -1, 0.0_C_DOUBLE, error)
CALL assert(profile == 0.0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)

profile = ComptonProfile_Partial(26, N2_SHELL, 0.0_C_DOUBLE)
CALL assert(profile == 0.0)
profile = ComptonProfile_Partial(26, N2_SHELL, 0.0_C_DOUBLE, error)
CALL assert(profile == 0.0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
DEALLOCATE(error)


ENDPROGRAM test_comptonprofiles
