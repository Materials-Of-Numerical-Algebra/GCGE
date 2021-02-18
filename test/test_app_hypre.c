#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "ops.h"
#include "app_hypre.h"

#define USE_PHG_MAT 1
/* run this program using the console pauser or add your own getch, system("pause") or input loop */
int TestVec              (void *mat, struct OPS_ *ops);
int TestMultiVec         (void *mat, struct OPS_ *ops);
int TestOrth             (void *mat, struct OPS_ *ops);
int TestLinearSolver     (void *mat, struct OPS_ *ops);
int TestMultiLinearSolver(void *mat, struct OPS_ *ops);
int TestEigenSolver      (void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);
int TestMultiGrid        (void *A, void *B, struct OPS_ *ops);


#if USE_HYPRE
int  CreateMatrixHYPRE (HYPRE_IJMatrix *A, HYPRE_IJMatrix *B, HYPRE_IJVector *x, int argc, char *argv[]);
int  DestroyMatrixHYPRE(HYPRE_IJMatrix *A, HYPRE_IJMatrix *B, HYPRE_IJVector *x, int argc, char *argv[]);

void MatrixConvertPHG2HYPRE(void **hypre_mat,  void **phg_mat);
int  CreateMatrixPHG (void **matA, void **matB, void **dofU, void **mapM, void **gridG, int argc, char *argv[]);
int  DestroyMatrixPHG(void **matA, void **matB, void **dofU, void **mapM, void **gridG, int argc, char *argv[]);


/*
  Create an application context to contain data needed by the
  application-provided call-back routines, ops->MultiLinearSolver().
*/
typedef struct {
   HYPRE_Solver hypre_solver;
} AppCtx;

static void AppCtxCreate(AppCtx *user, hypre_ParCSRMatrix *hypre_mat, hypre_ParVector *hypre_vec)
{
   HYPRE_BoomerAMGCreate(&(user->hypre_solver));
   //HYPRE_BoomerAMGSetMaxLevels(user->hypre_solver, 4);
   //HYPRE_BoomerAMGSetStrongThreshold(user->hypre_solver, 0.25);

   HYPRE_BoomerAMGSetCoarsenType(user->hypre_solver, 6);
   HYPRE_BoomerAMGSetInterpType(user->hypre_solver, 0);
   HYPRE_BoomerAMGSetPMaxElmts(user->hypre_solver, 0);

   HYPRE_BoomerAMGSetMaxIter(user->hypre_solver, 1);
   HYPRE_BoomerAMGSetTol(user->hypre_solver, 0.0);

   HYPRE_BoomerAMGSetNumSweeps(user->hypre_solver, 1);
   HYPRE_BoomerAMGSetRelaxOrder(user->hypre_solver, 1);
   HYPRE_BoomerAMGSetRelaxType(user->hypre_solver, 6);

   HYPRE_BoomerAMGSetPrintLevel(user->hypre_solver, 3);
   HYPRE_BoomerAMGSetup(user->hypre_solver, hypre_mat, hypre_vec, hypre_vec);
   return;
}
static void AppCtxDestroy(AppCtx *user)
{
   HYPRE_BoomerAMGDestroy(user->hypre_solver);
   return;
}
void HYPRE_MultiLinearSolver(void *mat, void **b, void **x, int *start, int *end, struct OPS_ *ops)
{
   ops->Printf("HYPRE_MultiLinearSolver\n");
   assert(end[0]-start[0]==end[1]-start[1]);
   AppCtx *user = (AppCtx*)ops->multi_linear_solver_workspace;
   hypre_ParVector* hypre_vec_b = (hypre_ParVector*)b;
   hypre_ParVector* hypre_vec_x = (hypre_ParVector*)x;
   int     nvec_b = hypre_vec_b->local_vector->num_vectors,  nvec_x = hypre_vec_x->local_vector->num_vectors;
   double *data_b = hypre_vec_b->local_vector->data       , *data_x = hypre_vec_x->local_vector->data;
   hypre_vec_b->local_vector->num_vectors = 1; 
   hypre_vec_x->local_vector->num_vectors = 1;
   hypre_vec_b->local_vector->data += start[0]*hypre_vec_b->local_vector->size;
   hypre_vec_x->local_vector->data += start[1]*hypre_vec_x->local_vector->size;
   int idx, ncols = end[0]-start[0];
   for (idx = 0; idx < ncols; ++idx) {
      HYPRE_BoomerAMGSolve(user->hypre_solver, 
	    (hypre_ParCSRMatrix*)mat, hypre_vec_b, hypre_vec_x);
      hypre_vec_b->local_vector->data += hypre_vec_b->local_vector->size;
      hypre_vec_x->local_vector->data += hypre_vec_x->local_vector->size;
      /* Run info - needed logging turned on */
#if DEBUG
      int num_iterations; double final_res_norm;
      HYPRE_BoomerAMGGetNumIterations(user->hypre_solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(user->hypre_solver, &final_res_norm);
      ops->Printf("[%d] Iterations = %d\n", idx+start[1], num_iterations);
      ops->Printf("[%d] Final Relative Residual Norm = %e\n", idx+start[1], final_res_norm);
#endif
   }
   hypre_vec_b->local_vector->num_vectors = nvec_b; hypre_vec_x->local_vector->num_vectors = nvec_x;
   hypre_vec_b->local_vector->data        = data_b; hypre_vec_x->local_vector->data        = data_x;
   ops->Printf("HYPRE_MultiLinearSolver\n");
   return;
}


static char help[] = "Test App of HYPRE.\n";
int TestAppHYPRE(int argc, char *argv[]) 
{
   MPI_Init(&argc, &argv);
   void *matA, *matB, *vecX; OPS *ops;
   OPS *hypre_ops = NULL;
   OPS_Create (&hypre_ops);
   OPS_HYPRE_Set (hypre_ops);
   OPS_Setup (hypre_ops);
   hypre_ops->Printf("%s", help);
   HYPRE_IJMatrix hypre_mat_A, hypre_mat_B;
#if USE_PHG_MAT
   void *phg_mat_A, *phg_mat_B, *phg_dof_U, *phg_map_M, *phg_grid_G;
   CreateMatrixPHG (&phg_mat_A, &phg_mat_B, &phg_dof_U, &phg_map_M, &phg_grid_G, argc, argv);
   MatrixConvertPHG2HYPRE((void **)(&hypre_mat_A), &phg_mat_A);
   MatrixConvertPHG2HYPRE((void **)(&hypre_mat_B), &phg_mat_B);
   DestroyMatrixPHG(&phg_mat_A, &phg_mat_B, &phg_dof_U, &phg_map_M, &phg_grid_G, argc, argv);
   HYPRE_IJMatrixGetObject(hypre_mat_A, &matA);
   HYPRE_IJMatrixGetObject(hypre_mat_B, &matB);
   hypre_ops->MultiVecCreateByMat((void ***)(&vecX), 1, matA, hypre_ops);
#else
   HYPRE_IJVector hypre_vec_x;
   CreateMatrixHYPRE (&hypre_mat_A, &hypre_mat_B, &hypre_vec_x, argc, argv);
   /* Get the parcsr matrix object to use */
   HYPRE_IJMatrixGetObject(hypre_mat_A, &matA);
   HYPRE_IJMatrixGetObject(hypre_mat_B, &matB);
   HYPRE_IJVectorGetObject(hypre_vec_x, &vecX);
#endif

   //hypre_ops->MatView((void*)matA, hypre_ops);
   //hypre_ops->MatView((void*)matB, hypre_ops);
   ops = hypre_ops;
   //TestMultiVec(matA,ops);
   //TestMultiLinearSolver(matA,ops);
   //TestOrth(matA,ops);

   /* flag == 1 表示使用外部线性求解器
    * flag == 2 表示使用外部线性求解器为Preconditioner */
   int flag = 0;
   AppCtx user; 
   if (flag>=1) {
      AppCtxCreate(&user, (hypre_ParCSRMatrix*)matA, (hypre_ParVector*)vecX);
      ops->multi_linear_solver_workspace = (void*)&user;
      ops->MultiLinearSolver = HYPRE_MultiLinearSolver;
   }
   TestEigenSolver(matA,matB,flag,argc,argv,ops);
   if (flag>=1) {
      AppCtxDestroy(&user);
   }
   //TestMultiGrid(matA,matB,ops);
   /* 销毁hypre矩阵 */
#if USE_PHG_MAT
   DestroyMatrixHYPRE(&hypre_mat_A, &hypre_mat_B, NULL, argc, argv);
   hypre_ops->MultiVecDestroy((void ***)(&vecX), 1, hypre_ops);
#else
   DestroyMatrixHYPRE(&hypre_mat_A, &hypre_mat_B, &hypre_vec_x, argc, argv);
#endif
   OPS_Destroy (&hypre_ops);
   MPI_Finalize();
   return 0;
}

int CreateMatrixHYPRE (HYPRE_IJMatrix *A, HYPRE_IJMatrix *B, HYPRE_IJVector *x, int argc, char *argv[])
{
   //int i, N, n = 3750;
   int i, N, n = 55;
   HYPRE_Int myid, num_procs;
   HYPRE_Int local_size, extra, ilower, iupper; 
   HYPRE_Real h;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   /* for 2D */
   /* Preliminaries: want at least one processor per row */
   if (n*n < num_procs) n = sqrt(num_procs) + 1;
   N = n*n; /* global number of rows */
   h = 1.0/(n+1); /* mesh size*/
   /* Each processor knows only of its own rows - the range is denoted by ilower
      and iupper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of processors. */
   local_size = N/num_procs;
   extra      = N - local_size*num_procs;
   ilower     = local_size*myid;
   ilower    += hypre_min(myid, extra);
   iupper     = local_size*(myid+1);
   iupper    += hypre_min(myid+1, extra);
   iupper     = iupper - 1;
   /* How many rows do I have? */
   local_size = iupper - ilower + 1;
   /* Create the matrix.
      Note that this is a square matrix, so we indicate the row partition
      size twice (since number of rows = number of cols) */
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, A);
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, B);
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, x);
   /* Choose a parallel csr format storage (see the User's Manual) */
   HYPRE_IJMatrixSetObjectType(*A, HYPRE_PARCSR);
   HYPRE_IJMatrixSetObjectType(*B, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(*x, HYPRE_PARCSR);
   /* Initialize before setting coefficients */
   HYPRE_IJMatrixInitialize(*A);
   HYPRE_IJMatrixInitialize(*B);
   HYPRE_IJVectorInitialize(*x);
   {
      int nnz;
      double values[5];
      int cols[5];
      for (i = ilower; i <= iupper; i++) {
	 nnz = 0;
	 /* The left identity block:position i-n */
	 if ((i-n)>=0) {
	    cols[nnz] = i-n;
	    values[nnz] = -1.0/h;
	    nnz++;
	 }
	 /* The left -1: position i-1 */
	 if (i%n) {
	    cols[nnz] = i-1;
	    values[nnz] = -1.0/h;
	    nnz++;
	 }
	 /* Set the diagonal: position i */
	 cols[nnz] = i;
	 values[nnz] = 4.0/h;
	 nnz++;
	 /* The right -1: position i+1 */
	 if ((i+1)%n) {
	    cols[nnz] = i+1;
	    values[nnz] = -1.0/h;
	    nnz++;
	 }
	 /* The right identity block:position i+n */
	 if ((i+n)< N) {
	    cols[nnz] = i+n;
	    values[nnz] = -1.0/h;
	    nnz++;
	 }
	 /* Set the values for row i */
	 HYPRE_IJMatrixSetValues(*A, 1, &nnz, &i, cols, values);
      }
   }
   {
      int nnz;
      double values[5];
      int cols[5];
      for (i = ilower; i <= iupper; i++) {
	 nnz = 1;
	 cols[0] = i;
  	 values[0] = 1.0*h;
	 /* Set the values for row i */
	 HYPRE_IJMatrixSetValues(*B, 1, &nnz, &i, cols, values);
      }
   }
   {
      double *x_values;
      int    *rows;
      x_values =  (double*) calloc(local_size, sizeof(double));
      rows = (int*) calloc(local_size, sizeof(int));
      for (i=0; i<local_size; i++)
      {
	 x_values[i] = 0.0;
	 rows[i] = ilower + i;
      }
      HYPRE_IJVectorSetValues(*x, local_size, rows, x_values);
      free(x_values);
      free(rows);
   }
   /* Assemble after setting the coefficients */
   HYPRE_IJMatrixAssemble(*A);
   HYPRE_IJMatrixAssemble(*B);
   HYPRE_IJVectorAssemble(*x);
   return 0;
}
int  DestroyMatrixHYPRE(HYPRE_IJMatrix *A, HYPRE_IJMatrix *B, HYPRE_IJVector *x, int argc, char *argv[])
{
   HYPRE_IJMatrixDestroy(*A);
   HYPRE_IJMatrixDestroy(*B);
   if (x!=NULL)
      HYPRE_IJVectorDestroy(*x);
   return 0;
}

#endif
