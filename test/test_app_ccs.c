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


int TestVec              (void *mat, struct OPS_ *ops);
int TestMultiVec         (void *mat, struct OPS_ *ops);
int TestOrth             (void *mat, struct OPS_ *ops);
int TestLinearSolver     (void *mat, struct OPS_ *ops);
int TestMultiLinearSolver(void *mat, struct OPS_ *ops);
int TestEigenSolverGCG   (void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);
int TestEigenSolverPAS   (void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);
int TestMultiGrid        (void *A, void *B, struct OPS_ *ops);


static int CreateMatrixCCS (CCSMAT *ccs_matA, CCSMAT *ccs_matB);
static int DestroyMatrixCCS(CCSMAT *ccs_matA, CCSMAT *ccs_matB);

#if OPS_USE_UMFPACK
#include "umfpack.h"
/*
  Create an application context to contain data needed by the
  application-provided call-back routines, ops->MultiLinearSolver().
*/
typedef struct {
   void   *Symbolic; 
   void   *Numeric;
   int    *Ap;
   int    *Ai;
   double *Ax;
   double *null;
   int    n;
} AppCtx;
static void AppCtxCreate(AppCtx *user, CCSMAT *ccs_mat)
{
   user->Ap   = ccs_mat->j_col;
   user->Ai   = ccs_mat->i_row;
   user->Ax   = ccs_mat->data;
   user->null = (double*)NULL;
   user->n    = ccs_mat->nrows;
   umfpack_di_symbolic(user->n, user->n, user->Ap, user->Ai, user->Ax, &(user->Symbolic), user->null, user->null);
   umfpack_di_numeric(user->Ap, user->Ai, user->Ax, user->Symbolic, &(user->Numeric), user->null, user->null);
   umfpack_di_free_symbolic(&(user->Symbolic));
   return;
}
static void AppCtxDestroy(AppCtx *user)
{
   umfpack_di_free_numeric(&(user->Numeric));
   user->Ap = NULL;
   user->Ai = NULL;
   user->Ax = NULL;
   return;
}
void UMFPACK_MultiLinearSolver(void *mat, void **b, void **x, int *start, int *end, struct OPS_ *ops)
{
   assert(end[0]-start[0]==end[1]-start[1]);
   AppCtx *user = (AppCtx*)ops->multi_linear_solver_workspace;
   LAPACKVEC *b_vec = (LAPACKVEC*)b, *x_vec = (LAPACKVEC*)x;
   double *b_array = b_vec->data+start[0]*b_vec->ldd; 
   double *x_array = x_vec->data+start[1]*x_vec->ldd; 
   int idx, ncols = end[0]-start[0];
   for (idx = 0; idx < ncols; ++idx) {
      umfpack_di_solve (UMFPACK_A, user->Ap, user->Ai, user->Ax, 
	    x_array, b_array, user->Numeric, user->null, user->null);
      b_array += b_vec->ldd; x_array += x_vec->ldd;
   }
   return;
}
#endif
int TestAppCCS(int argc, char *argv[]) 
{
#if OPS_USE_MPI
   MPI_Init(&argc, &argv);
#endif

   OPS *ccs_ops = NULL;
   OPS_Create (&ccs_ops);
   OPS_CCS_Set (ccs_ops);
   OPS_Setup (ccs_ops);

   void *matA, *matB; OPS *ops;

   CCSMAT ccs_matA, ccs_matB;
   CreateMatrixCCS(&ccs_matA, &ccs_matB);

   ops = ccs_ops; matA = (void*)(&ccs_matA); matB = (void*)(&ccs_matB);

   //TestMultiVec(matA,ops);
   //TestMultiLinearSolver(matA,ops);
   //TestOrth(matA,ops);
   /* The following three fucntions can not be test for PASMAT */
   //TestLinearSolver(matA,ops);
   /* flag == 0 表示不使用外部多向量线性求解器
    * flag == 1 表示仅使用外部多向量线性求解器
    * flag == 2 表示以外部多向量线性求解器为预条件子 */
   int flag = 0;
#if OPS_USE_UMFPACK 
   AppCtx user; flag = 1;
   if (flag>=1) {
      AppCtxCreate(&user, &ccs_matA);
      ops->multi_linear_solver_workspace = (void*)&user;
      ops->MultiLinearSolver = UMFPACK_MultiLinearSolver;
   }
#endif
   TestEigenSolverGCG(matA,matB,flag,argc,argv,ops);
#if OPS_USE_UMFPACK
   if (flag>=1) {
      AppCtxDestroy(&user);
   }
#endif
   //TestMultiGrid(matA,matB,ops);

   DestroyMatrixCCS(&ccs_matA, &ccs_matB);

   OPS_Destroy (&ccs_ops);
#if TEST_PAS
   OPS_Destroy (&pas_ops);
#endif

#if OPS_USE_MPI
   MPI_Finalize();
#endif
   return 0;
}

static int CreateMatrixCCS(CCSMAT *ccs_matA, CCSMAT *ccs_matB) 
{
   //int n = 12, row, col; double h = 1.0/(n+1);
   //int n = 800+7, col; double h = 1.0/(n+1);
   int n = 30000+7, col; double h = 1.0/(n+1);
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
