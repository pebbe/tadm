#include "tadm.h"
PetscErrorCode FIX_noerror()
{
  return 0;
}

PetscErrorCode MatCreateMPIAIJ(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], Mat *A)
{
  PetscMPIInt size;

  MatCreate(comm, A);
  MatSetSizes(*A, m, n, M, N);
  MPI_Comm_size(comm, &size);
  if (size > 1)
  {
    MatSetType(*A, MATMPIAIJPERM);
    MatMPIAIJSetPreallocation(*A, d_nz, d_nnz, o_nz, o_nnz);
  }
  else
  {
    MatSetType(*A, MATSEQAIJPERM);
    MatSeqAIJSetPreallocation(*A, d_nz, d_nnz);
  }
  return (0);
}
