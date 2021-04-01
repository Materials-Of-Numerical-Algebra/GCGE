/**
 *    @file  mat_convert.c
 *   @brief   
 *
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/9/13
 *      Revision:  none
 */

#include "ops.h"

#if OPS_USE_PHG & OPS_USE_PETSC & OPS_USE_HYPRE
#include "phg.h"

#include <petscmat.h>
#include <petscvec.h>

#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "_hypre_parcsr_mv.h"

/**
 * @brief PHG MAT convert to HYPRE HYPRE_IJMatrix
 *
 * and Destroy PHG MAT, is it necessary ?
 *
 * @param hypre_ij_mat
 * @param phg_mat
 */
void MatrixConvertPHG2HYPRE(void **hypre_matrix, void **phg_matrix)
{
   phgPrintf ( "MatrixConvertPHG2HYPRE\n" );
   HYPRE_IJMatrix hypre_ij_mat;
   MAT *phg_mat   = (MAT *)(*phg_matrix);
   const MAT_ROW *row;
   MAP *map = phg_mat->rmap;
   int idx, j;
   //HYPRE_IJMatrixCreate(map->comm,
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD,
	 map->partition[map->rank],
	 map->partition[map->rank + 1] - 1,
	 map->partition[map->rank],
	 map->partition[map->rank + 1] - 1,
	 &hypre_ij_mat);
   HYPRE_IJMatrixSetObjectType(hypre_ij_mat,  HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(hypre_ij_mat);

   /* Put row j of phg mat into ij_mat of hypre */
   j = map->partition[map->rank];
   for (idx = 0; idx < map->nlocal; idx++,  j++)
   {
      row = phgMatGetRow(phg_mat,  idx);
      if (row->ncols <= 0)
	 phgError(1,  "%s: matrix row %d is empty!\n",  __func__,  idx);

      HYPRE_IJMatrixSetValues(hypre_ij_mat, 1, &row->ncols, &j, row->cols, row->data);

      if (phg_mat->refcount == 0 && phg_mat->rows != NULL)
      {
	 /*  free PHG matrix row */
	 phgFree(phg_mat->rows[idx].cols);
	 phgFree(phg_mat->rows[idx].data);
	 phg_mat->rows[idx].cols = NULL;
	 phg_mat->rows[idx].data = NULL;
	 phg_mat->rows[idx].ncols = phg_mat->rows[idx].alloc = 0;
      }
   }
   HYPRE_IJMatrixAssemble(hypre_ij_mat);

   //if (phg_mat->refcount == 0)
   //  phgMatFreeMatrix(phg_mat);

   *hypre_matrix = (void *)(hypre_ij_mat);

   phgPrintf ( "MatrixConvertPHG2HYPRE\n" );
   return;
}





/* MPI的函数被PHG重新定义 */
void MatrixConvertPHG2PETSC(void **petsc_matrix, void **phg_matrix)
{
   phgPrintf ( "MatrixConvertPHG2PETSC\n" );
   Mat petsc_mat;
   MAT *phg_mat   = (MAT *)(*phg_matrix);
   const MAT_ROW *row;
   MAP *map = phg_mat->rmap;
   int idx, j;

if (1) {
   PetscInt N = map->nglobal, Nlocal = map->nlocal;
   PetscInt i, prealloc, *d_nnz, *o_nnz;
   INT d, o;
   /* CSR matrix */
   prealloc = PETSC_DECIDE;
   d_nnz = phgAlloc(Nlocal * sizeof(*d_nnz));
   o_nnz = phgAlloc(Nlocal * sizeof(*o_nnz));
   for (i = 0; i < Nlocal; i++) {
   	phgMatGetNnzInRow(phg_mat, i, &d, &o);
   	d_nnz[i] = d;
   	o_nnz[i] = o;
   }
   MatCreateAIJ(map->comm, Nlocal, Nlocal, N, N,
		   prealloc, d_nnz, prealloc, o_nnz, &petsc_mat);
   phgFree(d_nnz);
   phgFree(o_nnz);
}
else {
   MatCreate(PETSC_COMM_WORLD, &petsc_mat);
   //printf ( "[%d]: from %d to %d (%d)\n", 
   //	 map->rank,
   //	 map->partition[map->rank], map->partition[map->rank+1], 
   //	 map->nglobal);
   MatSetSizes(petsc_mat, 
	 map->partition[map->rank+1] - map->partition[map->rank], 
	 map->partition[map->rank+1] - map->partition[map->rank], 
	 map->nglobal, map->nglobal);
   MatSetType(petsc_mat, MATAIJ);
   MatSetUp(petsc_mat);
}




   /* Put row j of phg mat into Mat of petsc */
   j = map->partition[map->rank];
   for (idx = 0; idx < map->nlocal; idx++,  j++) 
   {
      row = phgMatGetRow(phg_mat,  idx);
      if (row->ncols <= 0)
	 phgError(1,  "%s: matrix row %d is empty!\n",  __func__,  idx);

      //printf ( "j = %d, row->ncols = %d\n", j, row->ncols );
      MatSetValues(petsc_mat, 1, &j, row->ncols, row->cols, row->data, INSERT_VALUES); 

      if (phg_mat->refcount == 0 && phg_mat->rows != NULL) 
      {
	 /*  free PHG matrix row */
	 phgFree(phg_mat->rows[idx].cols);
	 phgFree(phg_mat->rows[idx].data);
	 phg_mat->rows[idx].cols = NULL;
	 phg_mat->rows[idx].data = NULL;
	 phg_mat->rows[idx].ncols = phg_mat->rows[idx].alloc = 0;
      }
   }
   MatAssemblyBegin(petsc_mat, MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(petsc_mat, MAT_FINAL_ASSEMBLY);

   //int nrows, ncols;
   //MatGetLocalSize(petsc_mat, &nrows, &ncols);
   //PetscPrintf(PETSC_COMM_SELF, "[%d]: %d, %d\n", map->rank, nrows, ncols);


//   if (phg_mat->refcount == 0)
//      phgMatFreeMatrix(phg_mat);

   *petsc_matrix = (void *)(petsc_mat);
   //MatView((Mat)(*petsc_matrix), PETSC_VIEWER_STDOUT_WORLD);
   phgPrintf ( "MatrixConvertPHG2PETSC\n" );
   return;
}
#endif
