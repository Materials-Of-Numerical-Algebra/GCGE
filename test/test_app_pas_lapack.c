/**
 *    @file  test_app_lapack.c
 *   @brief  test app of LAPACK
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

#include "ops.h"
#include "app_lapack.h"
#include "app_pas.h"
int TestVec              (void *mat, struct OPS_ *ops);
int TestMultiVec         (void *mat, struct OPS_ *ops);
int TestOrth             (void *mat, struct OPS_ *ops);
int TestLinearSolver     (void *mat, struct OPS_ *ops);
int TestMultiLinearSolver(void *mat, struct OPS_ *ops);
int TestEigenSolverGCG   (void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);


int TestAppPAS_LAPACK(int argc, char *argv[]) 
{
#if USE_MPI
   MPI_Init(&argc, &argv);
#endif
	
	OPS *lapack_ops = NULL;
	OPS_Create (&lapack_ops);
	OPS_LAPACK_Set (lapack_ops);
	OPS_Setup (lapack_ops);

	OPS *pas_ops = NULL;
	OPS_Create (&pas_ops);
	OPS_PAS_Set (pas_ops,lapack_ops);
	OPS_Setup (pas_ops);
	void *matA, *matB; OPS *ops;

	int n = 50, row, col; double h = 1.0/(n+1);
	//int n = 1000+7, row, col; double h = 1.0/(n+1);
	LAPACKMAT lapack_matA; 
	lapack_matA.nrows = n; lapack_matA.ncols = n; lapack_matA.ldd = n;
	lapack_matA.data  = malloc(n*n*sizeof(double));
	for (col = 0; col < n; ++col) {
		for (row = 0; row < n; ++row) {			
			if (row == col) lapack_matA.data[row+n*col] = 2.0/(h);
			else if (row-col == 1) lapack_matA.data[row+n*col] = -1.0/(h);
			else if (row-col ==-1) lapack_matA.data[row+n*col] = -1.0/(h);
			else lapack_matA.data[row+n*col] = 0.0;
		}			
	}
	//for (col = 0; col < n; ++col) {
	//	for (row = 0; row < n; ++row) {			
	//		if (row == col) lapack_matA.data[row+n*col] = row+col+1;
	//		else lapack_matA.data[row+n*col] = 0.0;
	//	}			
	//}

	LAPACKMAT lapack_matB;
	lapack_matB.nrows = n; lapack_matB.ncols = n; lapack_matB.ldd = n;
	lapack_matB.data  = malloc(n*n*sizeof(double));
	for (col = 0; col < n; ++col) {		
		for (row = 0; row < n; ++row) {
			if (row == col) lapack_matB.data[row+n*col] = 1.0*h;
			else lapack_matB.data[row+n*col] = 0.0;
		}
	}

	LAPACKVEC lapack_vec;
	lapack_vec.nrows = n; lapack_vec.ncols = n; lapack_vec.ldd = n;
	lapack_vec.data  = malloc(n*n*sizeof(double));
	for (col = 0; col < n; ++col) {		
		for (row = 0; row < n; ++row) {
			if (row == col) lapack_vec.data[row+n*col] = -0.1;
			else lapack_vec.data[row+n*col] = 0.0;
		}
	}
	
	PASMAT pas_matA, pas_matB; 
	pas_matA.level_aux = 0; pas_matA.num_levels = 1;
	pas_matA.QQ = malloc(sizeof(void*));
	pas_matA.QX = malloc(sizeof(void**)); 
	pas_matA.XX    = (void*)&lapack_matA;
	pas_matA.QQ[0] = (void*)&lapack_matA;
	pas_matA.QX[0] = (void**)&lapack_vec;
	
	/* pas_matB.XX 每一层都是 单位矩阵, QX 每一层都是 零向量组 */
	pas_matB.level_aux = 0; pas_matB.num_levels = 1;
	pas_matB.QQ = malloc(sizeof(void*));
	pas_matB.QX = NULL; 
	pas_matB.XX = NULL;
	pas_matB.QQ[0] = (void*)&lapack_matB;

	ops = pas_ops; matA = &pas_matA; matB = &pas_matB;
	
	//TestMultiVec(matA,ops);
	TestMultiLinearSolver(matA,ops);
	//TestOrth(matA,ops);
	//TestEigenSolverGCG(matA,matB,0,argc,argv,ops);

 	free(pas_matA.QQ); pas_matA.QQ = NULL;
 	free(pas_matA.QX); pas_matA.QX = NULL;
 	free(pas_matB.QQ); pas_matB.QQ = NULL;
 
 	free(lapack_vec.data) ; lapack_vec.data  = NULL;

	free(lapack_matA.data); lapack_matA.data = NULL;
	free(lapack_matB.data); lapack_matB.data = NULL;
	
	OPS_Destroy (&lapack_ops);
	OPS_Destroy (&pas_ops);
	
#if USE_MPI
   MPI_Finalize();
#endif
	return 0;
}

