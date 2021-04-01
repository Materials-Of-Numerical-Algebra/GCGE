#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "ops.h"
#include "app_phg.h"
#include "app_pas.h"

#if OPS_USE_MUMPS
#include "dmumps_c.h"
#define ICNTL(I) icntl[(I)-1] 
#define INFOG(I) infog[(I)-1] 
#define CNTL(I)  cntl[(I)-1] 
#define INFO(I)  info[(I)-1]
#endif
/* run this program using the console pauser or add your own getch, system("pause") or input loop */
int TestVec              (void *mat, struct OPS_ *ops);
int TestMultiVec         (void *mat, struct OPS_ *ops);
int TestOrth             (void *mat, struct OPS_ *ops);
int TestLinearSolver     (void *mat, struct OPS_ *ops);
int TestMultiLinearSolver(void *mat, struct OPS_ *ops);
int TestEigenSolverGCG   (void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);
int TestEigenSolverPAS   (void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);
int TestMultiGrid        (void *A, void *B, struct OPS_ *ops);


#if OPS_USE_PHG
int  CreateMatrixPHG (void **matA, void **matB, void **dofU, void **mapM, void **gridG, int argc, char *argv[]);
int  DestroyMatrixPHG(void **matA, void **matB, void **dofU, void **mapM, void **gridG, int argc, char *argv[]);

#if OPS_USE_MUMPS
/*
  Create an application context to contain data needed by the
  application-provided call-back routines, ops->MultiLinearSolver().
*/
typedef struct {
	DMUMPS_STRUC_C mumps_solver;
	int n;
	int nnz_local;
	int *rows;
	int *cols;
	double *values;
	double *sol;
} AppCtx;

static void AppCtxCreate(AppCtx *user, MAT *phg_mat)
{
	double time_start, time_end; 
	time_start = MPI_Wtime();
	int row, j, nnz_local, *pc;

	user->mumps_solver.comm_fortran = MPI_Comm_c2f(phg_mat->rmap->comm);
	user->mumps_solver.par =  1;
	user->mumps_solver.sym =  0; /* unsymmetric.*/
	user->mumps_solver.job = -1; /* initializes an instance of the package */
	dmumps_c(&(user->mumps_solver));

	nnz_local = phg_mat->nnz_d + phg_mat->nnz_o;

	user->n         = phg_mat->rmap->nglobal;
	user->nnz_local = nnz_local;
	user->rows      = malloc(nnz_local*sizeof(int));
	user->cols      = malloc(nnz_local*sizeof(int));
	user->values    = phg_mat->packed_data;

	pc = phg_mat->packed_cols;
	for (row = 0; row < phg_mat->rmap->nlocal; ++row) {
		for (j = phg_mat->packed_ind[row]; j < phg_mat->packed_ind[row+1]; ++j) {
			/* in-process entry */
			user->rows[j] = 1+row   + phg_mat->rmap->partition[phg_mat->rmap->rank];
			user->cols[j] = 1+pc[j] + phg_mat->rmap->partition[phg_mat->rmap->rank];
		}
		for (j = phg_mat->packed_ind[row+phg_mat->rmap->nlocal]; 
				j < phg_mat->packed_ind[row+phg_mat->rmap->nlocal+1]; ++j) {
			/* off-process entry */
			user->rows[j] = 1+row + phg_mat->rmap->partition[phg_mat->rmap->rank];
			user->cols[j] = 1+phg_mat->O2Gmap[phg_mat->ordering[ pc[j] ]];;
		}
	}
	user->mumps_solver.n       = user->n;
	user->mumps_solver.nnz_loc = user->nnz_local;
	user->mumps_solver.irn_loc = user->rows;
	user->mumps_solver.jcn_loc = user->cols;
	user->mumps_solver.a_loc   = user->values;
#if 0
	user->mumps_solver.ICNTL(1)  =  6; /* the output stream for error messages */
	user->mumps_solver.ICNTL(2)  =  1; /* the output stream for diagnostic printing and statistics local to each MPI process */
	user->mumps_solver.ICNTL(3)  =  6; /* the output stream for global information */
	user->mumps_solver.ICNTL(4)  =  4; /* errors, warnings and information on input, output parameters printed.*/
#else
	user->mumps_solver.ICNTL(1)  = -1; /* the output stream for error messages */
	user->mumps_solver.ICNTL(2)  = -1; /* the output stream for diagnostic printing and statistics local to each MPI process */
	user->mumps_solver.ICNTL(3)  = -1; /* the output stream for global information */
	user->mumps_solver.ICNTL(4)  =  0; /* errors, warnings and information on input, output parameters printed.*/
#endif
	user->mumps_solver.ICNTL(5)  =  0; /* assembled format */
	user->mumps_solver.ICNTL(9)  =  1; /* AX = B is solved */
	user->mumps_solver.ICNTL(10) =  0; /* maximum number of steps of iterative refinement */
  	user->mumps_solver.CNTL(2)   =0.0; /* stopping criterion for iterative refinement */
	user->mumps_solver.ICNTL(18) =  3; /* the distributed matrix */
	user->mumps_solver.ICNTL(20) =  0; /* the dense format of the right-hand side */
	user->mumps_solver.ICNTL(21) =  0; /* the centralized format of the solution */
	/* factorization */
	user->mumps_solver.job = 4; /* perform the analysis and the factorization */
	dmumps_c(&(user->mumps_solver));
	if (user->mumps_solver.INFOG(1)<0)
		printf("\n (PROC %d) ERROR RETURN: \tINFOG(1)= %d\n\t\t\t\tINFOG(2)= %d\n",
				phg_mat->rmap->rank, user->mumps_solver.INFOG(1), user->mumps_solver.INFOG(2));

	if (phg_mat->rmap->rank == 0) {
		/* 160 表示求解时, 至多160个向量一起算, blockSize<=160 */
		user->sol = malloc(160*user->n*sizeof(double));
	}
	else {
		user->sol = NULL;
	}
	time_end = MPI_Wtime();
	if (phg_mat->rmap->rank == 0) {
		printf("FACTORIZATION time %f\n", time_end-time_start);
	}
#if 0
if (phg_mat->cmap->rank == 0) {
	printf("%d,%d,%d,%d\n",sizeof(int),sizeof(MUMPS_INT),sizeof(MUMPS_INT8),sizeof(double));
	printf("%d n = %d, nnz_loc = %ld, nnz_local = %d\n",phg_mat->rmap->rank, user->mumps_solver.n, user->mumps_solver.nnz_loc,nnz_local);
	printf("%d nlocal = %d, nglobal = %d\n",phg_mat->rmap->rank, phg_mat->rmap->nlocal, phg_mat->rmap->nglobal);
	for (j = 0; j < nnz_local; j++) {
		printf("%d (%d,%d) %.4e\n", phg_mat->cmap->rank,
				user->mumps_solver.irn_loc[j],user->mumps_solver.jcn_loc[j],user->mumps_solver.a_loc[j]);
	}
}
#endif

	return;
}
static void AppCtxDestroy(AppCtx *user)
{
	user->mumps_solver.job = -2; /* terminates an instance of the package */
	dmumps_c(&(user->mumps_solver));
	if (user->sol!=NULL) free(user->sol);
	user->sol = NULL;
	free(user->rows); free(user->cols);
	user->rows = NULL; user->cols = NULL;
	return;
}
void MUMPS_MultiLinearSolver(void *mat, void **b, void **x, int *start, int *end, struct OPS_ *ops)
{
   ops->Printf("MUMPS_MultiLinearSolver\n");
   assert(end[0]-start[0]==end[1]-start[1]);
   int nvec = end[0]-start[0];
   double *data_b, *data_x;
   AppCtx *user = (AppCtx*)ops->multi_linear_solver_workspace;
   MAT *phg_mat   = (MAT*)mat;
   VEC *phg_vec_b = (VEC*)b;
   VEC *phg_vec_x = (VEC*)x;
   data_b = phg_vec_b->data; data_x = phg_vec_x->data;
   phg_vec_b->data += start[0]*phg_vec_b->map->nlocal;
   phg_vec_x->data += start[1]*phg_vec_x->map->nlocal;

   MAP *map = phg_mat->rmap;
   int nlocal = map->nlocal;
   int *cnts = phgAlloc(2 * map->nprocs * sizeof(*cnts));
   int *dsps = cnts + map->nprocs;
   int i;

   double time_start, time_end; 
   time_start = MPI_Wtime();
   for (i = 0; i < map->nprocs; i++) {
	   cnts[i] = map->partition[i + 1] - map->partition[i];
	   dsps[i] = map->partition[i];
   }
   MPI_Datatype *rowType = malloc(map->nprocs*sizeof(MPI_Datatype)); 
   MPI_Request  *request = malloc(map->nprocs*sizeof(MPI_Request ));
   for (i = 0; i < map->nprocs; ++i) {
	   MPI_Type_vector(nvec, cnts[i], user->n, MPI_DOUBLE, rowType+i);
	   MPI_Type_commit(rowType+i);
   }

   MPI_Isend(phg_vec_b->data, nvec*cnts[map->rank], MPI_DOUBLE, 0, map->rank, map->comm, request+map->rank);
   if (map->rank == 0) {
	   for (i = 0; i < map->nprocs; ++i) {
		   MPI_Irecv(user->sol+dsps[i], 1, rowType[i], i, i, map->comm, request+i);
	   }
	   for (i = 0; i < map->nprocs; ++i) {
		   MPI_Wait(request+i,MPI_STATUS_IGNORE);
	   }
   }
   else {
	   MPI_Wait(request+map->rank,MPI_STATUS_IGNORE);
   }
   time_end = MPI_Wtime();
   ops->Printf("GATHER time %f\n", time_end-time_start);

#if 0
   int k;
   if (phg_mat->cmap->rank == 0) {
	   printf("==========rhs========\n");
	   for (k = 0; k < user->n; ++k) {
		   printf("%.4e\n", user->sol[k]);
	   }
   }
#endif
   time_start = MPI_Wtime();

   user->mumps_solver.nrhs = nvec;
   user->mumps_solver.lrhs = user->n;
   user->mumps_solver.rhs  = user->sol;
   user->mumps_solver.job  = 3;
   dmumps_c(&(user->mumps_solver));
   if (user->mumps_solver.infog[0]<0)
	   printf("\n (PROC %d) ERROR RETURN: \tINFOG(1)= %d\n\t\t\t\tINFOG(2)= %d\n",
			   phg_mat->rmap->rank, user->mumps_solver.INFOG(1), user->mumps_solver.INFOG(2));
#if 0
   if (phg_mat->cmap->rank == 0) {
	   printf("==========sol========\n");
	   for (k = 0; k < user->n; ++k) {
		   printf("%.4e\n", user->sol[k]);
	   }
   }
#endif
   time_end = MPI_Wtime();
   ops->Printf("CALCULATE time %f\n", time_end-time_start);

   time_start = MPI_Wtime();
   if (map->rank == 0) {
	   for (i = 0; i < map->nprocs; ++i) {
		   MPI_Isend(user->sol+dsps[i], 1, rowType[i], i, i, map->comm, request+i);
	   }
   }
   MPI_Irecv(phg_vec_x->data, nvec*cnts[map->rank], MPI_DOUBLE, 0, map->rank, map->comm, request+map->rank);
   if (map->rank == 0) {
	   for (i = 0; i < map->nprocs; ++i) {
		   MPI_Wait(request+i,MPI_STATUS_IGNORE);
	   }
   }
   else {
	   MPI_Wait(request+map->rank,MPI_STATUS_IGNORE);
   }

   for (i = 0; i < map->nprocs; ++i) {
	   MPI_Type_free(rowType+i);
   }
   free(rowType);
   free(request);

   time_end = MPI_Wtime();
   ops->Printf("SCATTER time %f\n", time_end-time_start);

#if 0
   if (phg_mat->cmap->rank == 0) {
	   printf("==========x========\n");
	   for (k = 0; k < nlocal; ++k) {
		   printf("%.4e\n", phg_vec_x->data[k]);
	   }
   }
#endif

   phg_vec_b->data = data_b; phg_vec_x->data = data_x;


   ops->Printf("MUMPS_MultiLinearSolver\n");
   return;
}
#endif

static char help[] = "Test App of PHG.\n";
int TestAppPHG(int argc, char *argv[]) 
{
   
   //phgInit(&argc, &argv);
   void *phg_mat_A, *phg_mat_B, *phg_dof_U, *phg_map_M, *phg_grid_G;
   CreateMatrixPHG (&phg_mat_A, &phg_mat_B, &phg_dof_U, &phg_map_M, &phg_grid_G, argc, argv);

   void *matA, *matB; OPS *ops;
   OPS *phg_ops = NULL;
   OPS_Create (&phg_ops);
   OPS_PHG_Set (phg_ops);
   OPS_Setup (phg_ops);
   phg_ops->Printf("%s",help);

   //phg_ops->MatView((void*)phg_matA, phg_ops);
   //phg_ops->MatView((void*)phg_matB, phg_ops);
   ops = phg_ops; matA = (void*)(phg_mat_A); matB = (void*)(phg_mat_B);

   //TestMultiVec(matA,ops);
   //TestMultiLinearSolver(matA,ops);
   //TestOrth(matB,ops);

   int flag = 0;
#if OPS_USE_MUMPS
   AppCtx user;  /* VERY BAD efficiency */
   if (flag>=1) {
      AppCtxCreate(&user, (MAT*)matA);
      ops->multi_linear_solver_workspace = (void*)&user;
      ops->MultiLinearSolver = MUMPS_MultiLinearSolver;
   }
#endif
   TestEigenSolverGCG(matA,matB,flag,argc,argv,ops);
#if OPS_USE_MUMPS
   if (flag>=1) {
      AppCtxDestroy(&user);
   }
#endif
   //TestEigenSolverGCG(matA,matB,0,argc,argv,ops);
   //TestMultiGrid(matA,matB,ops);
   /* 销毁phg矩阵 */
   OPS_Destroy (&phg_ops);

   DestroyMatrixPHG(&phg_mat_A, &phg_mat_B, &phg_dof_U, &phg_map_M, &phg_grid_G, argc, argv);
   //phgFinalize();
   return 0;
}
#endif
