/**
 *    @file  test_app_ccs.c
 *   @brief  test app of CCS 
 *
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/17
 *      Revision:  none
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>


#include "ops.h"
#include "app_ccs.h"
#include "app_pas.h"


int TestMultiVec         (void *mat, struct OPS_ *ops);
int TestOrth             (void *mat, struct OPS_ *ops);
int TestMultiLinearSolver(void *mat, struct OPS_ *ops);
int TestEigenSolverGCG   (void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);

static int CreateMatrixCCS (CCSMAT *ccs_matA, CCSMAT *ccs_matB);
static int DestroyMatrixCCS(CCSMAT *ccs_matA, CCSMAT *ccs_matB);

int TestAppPAS_CCS(int argc, char *argv[]) 
{
#if USE_MPI
   MPI_Init(&argc, &argv);
#endif

   OPS *ccs_ops = NULL;
   OPS_Create (&ccs_ops);
   OPS_CCS_Set (ccs_ops);
   OPS_Setup (ccs_ops);

   OPS *pas_ops = NULL;
   OPS_Create (&pas_ops);
   OPS_PAS_Set (pas_ops,ccs_ops);
   OPS_Setup (pas_ops);

   void *matA, *matB; OPS *ops;

   CCSMAT ccs_matA, ccs_matB;
   CreateMatrixCCS(&ccs_matA, &ccs_matB);

   int n = ccs_matA.nrows, row, col; double h = 1.0/(n+1);
   LAPACKMAT lapack_matA; 
   lapack_matA.nrows = n; lapack_matA.ncols = n; lapack_matA.ldd = n;
   lapack_matA.data  = malloc(n*n*sizeof(double));
   for (col = 0; col < n; ++col) {
	   for (row = 0; row < n; ++row) {			
		   if (row == col) lapack_matA.data[row+n*col] = 2.0/h;
		   else if (row-col == 1) lapack_matA.data[row+n*col] = -1.0/h;
		   else if (row-col ==-1) lapack_matA.data[row+n*col] = -1.0/h;
		   else lapack_matA.data[row+n*col] = 0.0;
	   }			
   }

   LAPACKVEC lapack_vec;
   lapack_vec.nrows = n; lapack_vec.ncols = n; lapack_vec.ldd = n;
   lapack_vec.data  = malloc(n*n*sizeof(double));
   for (col = 0; col < n; ++col) {		
	   for (row = 0; row < n; ++row) {
		   if (row == col) lapack_vec.data[row+n*row] = -0.1;
		   else lapack_vec.data[row+n*col] = 0.0;
	   }
   }
   PASMAT pas_matA, pas_matB; 
   pas_matA.level_aux = 0; pas_matA.num_levels = 1;
   pas_matA.QQ = malloc(sizeof(void*));
   pas_matA.QX = malloc(sizeof(void**)); 
   pas_matA.XX    = (void*)&lapack_matA;
   pas_matA.QQ[0] = (void*)&ccs_matA;
   pas_matA.QX[0] = (void**)&lapack_vec;
   pas_matB.level_aux = 0; pas_matB.num_levels = 1;
   pas_matB.QQ = malloc(sizeof(void*));
   pas_matB.QX = NULL; 
   pas_matB.XX = NULL;
   pas_matB.QQ[0] = (void*)&ccs_matB;

   ops = pas_ops; matA = &pas_matA; matB = &pas_matB;


   //TestMultiVec(matA,ops);
   //TestMultiLinearSolver(matA,ops);
   //TestOrth(matA,ops);
   /* flag == 0 表示不使用外部多向量线性求解器
    * flag == 1 表示仅使用外部多向量线性求解器
    * flag == 2 表示以外部多向量线性求解器为预条件子 */
   /* 考虑使用 PAS 自带的线性解法器, 同时使用 UMFPACK
    * 值得注意的是, GCG 的线性解法器只是为了得到新的向量, 而非求解线性方程组 */
   int flag = 0;
   TestEigenSolverGCG(matA,matB,flag,argc,argv,ops);

   free(pas_matA.QQ); pas_matA.QQ = NULL;
   free(pas_matA.QX); pas_matA.QX = NULL;
   free(pas_matB.QQ); pas_matB.QQ = NULL;
   free(pas_matB.QX); pas_matB.QX = NULL;
   free(lapack_vec.data) ; lapack_vec.data  = NULL;
   DestroyMatrixCCS(&ccs_matA, &ccs_matB);

   OPS_Destroy (&ccs_ops);
   OPS_Destroy (&pas_ops);

#if USE_MPI
   MPI_Finalize();
#endif
   return 0;
}

static int CreateMatrixCCS(CCSMAT *ccs_matA, CCSMAT *ccs_matB) 
{
   int n = 50, row, col; double h = 1.0/(n+1);
   //int n = 1000+7, col; double h = 1.0/(n+1);
   ccs_matA->nrows = n; ccs_matA->ncols = n;
   ccs_matA->j_col = malloc((n+1)*sizeof(int));
   ccs_matA->i_row = malloc((3*n-2)*sizeof(int));
   ccs_matA->data  = malloc((3*n-2)*sizeof(double));

   ccs_matA->j_col[0] = 0; ccs_matA->j_col[1] = 2;
   ccs_matA->i_row[0] = 0; ccs_matA->i_row[1] = 1;
   ccs_matA->data[0]  = +2.0/h; 
   ccs_matA->data[1]  = -1.0/h;
   int idx = 2;
   for (col = 1; col < n-1; ++col) {
      ccs_matA->j_col[col+1] = ccs_matA->j_col[col]+3;
      ccs_matA->i_row[idx+0] = col-1;
      ccs_matA->i_row[idx+1] = col;
      ccs_matA->i_row[idx+2] = col+1;
      ccs_matA->data[idx+0]  = -1.0/h;
      ccs_matA->data[idx+1]  = +2.0/h;
      ccs_matA->data[idx+2]  = -1.0/h;
      idx += 3;
   }
   ccs_matA->j_col[n] = ccs_matA->j_col[n-1]+2;
   ccs_matA->i_row[3*n-4] = n-2; 
   ccs_matA->i_row[3*n-3] = n-1;
   ccs_matA->data[3*n-4]  = -1.0/h; 
   ccs_matA->data[3*n-3]  = +2.0/h;

   ccs_matB->nrows = n; ccs_matB->ncols = n;
   ccs_matB->j_col = malloc((n+1)*sizeof(int));
   ccs_matB->i_row = malloc(n*sizeof(int));
   ccs_matB->data  = malloc(n*sizeof(double));
   for (col = 0; col < n; ++col) {		
      ccs_matB->j_col[col] = col;
      ccs_matB->i_row[col] = col;
      ccs_matB->data[col]  = 1.0*h;
   }
   ccs_matB->j_col[n] = n;
   return 0;
}
static int DestroyMatrixCCS(CCSMAT *ccs_matA, CCSMAT *ccs_matB)
{
   free(ccs_matA->i_row); ccs_matA->i_row = NULL;
   free(ccs_matB->i_row); ccs_matB->i_row = NULL;
   free(ccs_matA->j_col); ccs_matA->j_col = NULL;
   free(ccs_matB->j_col); ccs_matB->j_col = NULL;
   free(ccs_matA->data) ; ccs_matA->data  = NULL;
   free(ccs_matB->data) ; ccs_matB->data  = NULL;
   return 0;
}
