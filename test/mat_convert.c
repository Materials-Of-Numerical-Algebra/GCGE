/*
 * =====================================================================================
 *
 *       Filename:  pase_convert.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年12月19日 09时57分13秒
 *
 *         Author:  Li Yu (liyu@tjufe.edu.cn), 
 *   Organization:  
 *
 * =====================================================================================
 */

#if USE_PHG && USE_SLEPC

#include <petscmat.h>
#include <petscvec.h>
#include "phg.h"

/* 进行测试时, 总有问题, MPI的函数被PHG重新定义 */
void MatrixConvertPHG2PETSC(void **petsc_matrix, void **phg_matrix)
{
   Mat petsc_mat;
   MAT *phg_mat   = (MAT *)(*phg_matrix);
   const MAT_ROW *row;
   MAP *map = phg_mat->rmap;
   int idx, j;

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
}

#endif
