#define MatDestroy(a) MatDestroy(&a)
#define MatTranspose(a, b) (b ? MatTranspose(a, MAT_INITIAL_MATRIX, b) : MatTranspose(a, MAT_REUSE_MATRIX, &a))

#define PetscGetTime(a) *a = 0
#define PetscLogEventRegister(a, b, c)    \
  {                                       \
    PetscLogEvent dummy;                  \
    PetscLogEventRegister(b, *a, &dummy); \
  }
#define PetscLogStageRegister(a, b) PetscLogStageRegister(b, a)
#define PetscOptionsGetInt(a, b, c, d) PetscOptionsGetInt(PETSC_NULL, a, b, c, d)
#define PetscOptionsGetReal(a, b, c, d) PetscOptionsGetReal(PETSC_NULL, a, b, c, d)
#define PetscOptionsGetString(a, b, c, d, e) PetscOptionsGetString(PETSC_NULL, a, b, c, d, e)
#define PetscOptionsHasName(a, b, c) PetscOptionsHasName(PETSC_NULL, a, b, c)
#define PetscOptionsLeft() PetscOptionsLeft(PETSC_NULL)
#define PetscSynchronizedFlush(a) PetscSynchronizedFlush(a, PETSC_STDOUT)
#define PetscViewerDestroy(a) PetscViewerDestroy(&a)
#define PetscTruth PetscBool

typedef struct TAO_APPLICATION
{
  int dummy;
} TAO_APPLICATION; /* TODO */
#define TAO_CONVERGED_TRTOL TAO_CONVERGED_STEPTOL
#define TAO_SOLVER Tao
#define TaoFinalize() FIX_noerror()
#define TaoGetIterationData(a,b,c,d,e,f,g)      TaoGetSolutionStatus(a,b,c,d,e,f,g)
#define TaoInitialize(a, b, c, d) /* TODO */
#define TaoRegisterDynamic(a, b, c, d) FIX_noerror()
#define TaoSetTerminationReason(a, b) TaoSetConvergedReason(a, b)
#define TaoTerminateReason TaoConvergedReason

#undef SETERRQ
#if defined(PETSC_USE_ERRORCHECKING)
#define SETERRQ(n, s) return PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, n, PETSC_ERROR_INITIAL, s)
#else // PETSC_USE_ERRORCHECKING
#define SETERRQ(n, s)
#endif // PETSC_USE_ERRORCHECKING

#define VecDestroy(a) VecDestroy(&a)

PetscErrorCode FIX_noerror();
PetscErrorCode MatCreateMPIAIJ(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], Mat *A);
