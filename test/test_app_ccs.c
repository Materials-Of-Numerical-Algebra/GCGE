#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ops.h"
#include "app_ccs.h"
#include "app_pas.h"
int TestVec              (void *mat, struct OPS_ *ops);
int TestMultiVec         (void *mat, struct OPS_ *ops);
int TestOrth             (void *mat, struct OPS_ *ops);
int TestLinearSolver     (void *mat, struct OPS_ *ops);
int TestMultiLinearSolver(void *mat, struct OPS_ *ops);
int TestEigenSolver      (void *A, void *B, int argc, char *argv[], struct OPS_ *ops);
int TestMultiGrid        (void *A, void *B, struct OPS_ *ops);

#define TEST_PAS 0

int TestAppCCS(int argc, char *argv[]) 
{
#if USE_MPI
	MPI_Init(&argc, &argv);
#endif
	
	OPS *ccs_ops = NULL;
	OPS_Create (&ccs_ops);
	OPS_CCS_Set (ccs_ops);
	OPS_Setup (ccs_ops);
#if TEST_PAS	
	OPS *pas_ops = NULL;
	OPS_Create (&pas_ops);
	OPS_PAS_Set (pas_ops,ccs_ops);
	OPS_Setup (pas_ops);
#endif	
	void *matA, *matB; OPS *ops;
	
	//int n = 12, row, col; double h = 1.0/(n+1);
	int n = 1000+7, row, col; double h = 1.0/(n+1);
	CCSMAT ccs_matA; 
	ccs_matA.nrows = n; ccs_matA.ncols = n;
	ccs_matA.j_col = malloc((n+1)*sizeof(int));
	ccs_matA.i_row = malloc((3*n-2)*sizeof(int));
	ccs_matA.data  = malloc((3*n-2)*sizeof(double));

	ccs_matA.j_col[0] = 0; ccs_matA.j_col[1] = 2;
	ccs_matA.i_row[0] = 0; ccs_matA.i_row[1] = 1;
	ccs_matA.data[0]  = +2.0/h; 
	ccs_matA.data[1]  = -1.0/h;
	int idx = 2;
	for (col = 1; col < n-1; ++col) {
		ccs_matA.j_col[col+1] = ccs_matA.j_col[col]+3;
		ccs_matA.i_row[idx+0] = col-1;
		ccs_matA.i_row[idx+1] = col;
		ccs_matA.i_row[idx+2] = col+1;
		ccs_matA.data[idx+0]  = -1.0/h;
		ccs_matA.data[idx+1]  = +2.0/h;
		ccs_matA.data[idx+2]  = -1.0/h;
		idx += 3;
	}
	ccs_matA.j_col[n] = ccs_matA.j_col[n-1]+2;
	ccs_matA.i_row[3*n-4] = n-2; 
	ccs_matA.i_row[3*n-3] = n-1;
	ccs_matA.data[3*n-4]  = -1.0/h; 
	ccs_matA.data[3*n-3]  = +2.0/h;

	CCSMAT ccs_matB;
	ccs_matB.nrows = n; ccs_matB.ncols = n;
	ccs_matB.j_col = malloc((n+1)*sizeof(int));
	ccs_matB.i_row = malloc(n*sizeof(int));
	ccs_matB.data  = malloc(n*sizeof(double));
	for (col = 0; col < n; ++col) {		
		ccs_matB.j_col[col] = col;
		ccs_matB.i_row[col] = col;
		ccs_matB.data[col]  = 1.0*h;
	}
	ccs_matB.j_col[n] = n;
#if TEST_PAS	
	LAPACKVEC lapack_vec;
	lapack_vec.nrows = n; lapack_vec.ncols = n; lapack_vec.ldd = n;
	lapack_vec.data  = malloc(n*n*sizeof(double));
	for (col = 0; col < n; ++col) {		
		for (row = 0; row < n; ++row) {
			if (row == col) lapack_vec.data[row+n*row] = 1.0;
			else lapack_vec.data[col+n*row] = 0.0;
		}
	}
	
	PASMAT pas_matA, pas_matB; 
	pas_matA.level_aux = 0; pas_matA.num_levels = 1;
	pas_matA.QQ = malloc(sizeof(void*));
	pas_matA.QX = malloc(sizeof(void**)); 
	pas_matA.XX    = (void*)&ccs_matA;
	pas_matA.QQ[0] = (void*)&ccs_matA;
	pas_matA.QX[0] = (void**)&lapack_vec;
	pas_matB.level_aux = 0; pas_matB.num_levels = 1;
	pas_matB.QQ = malloc(sizeof(void*));
	pas_matB.QX = malloc(sizeof(void**)); 
	pas_matB.XX    = (void*)&ccs_matB;
	pas_matB.QQ[0] = (void*)&ccs_matB;
	pas_matB.QX[0] = (void**)&lapack_vec;
#endif

#if TEST_PAS
	ops = pas_ops; matA = &pas_matA; matB = &pas_matB;
#else
	ops = ccs_ops; matA = (void*)(&ccs_matA); matB = (void*)(&ccs_matB);
#endif 
	
	//
	//TestMultiVec(matA,ops);
	//TestMultiLinearSolver(matA,ops);
	//TestOrth(matA,ops);
	/* The following three fucntions can not be test for PASMAT */
	//TestLinearSolver(matA,ops);
	TestEigenSolver(matA,matB,argc,argv,ops);
	//TestMultiGrid(matA,matB,ops);
#if TEST_PAS
 	free(pas_matA.QQ); pas_matA.QQ = NULL;
 	free(pas_matA.QX); pas_matA.QX = NULL;
 	free(pas_matB.QQ); pas_matB.QQ = NULL;
 	free(pas_matB.QX); pas_matB.QX = NULL;
 	free(lapack_vec.data) ; lapack_vec.data  = NULL;
#endif	
	free(ccs_matA.data); ccs_matA.data = NULL;
	free(ccs_matB.data); ccs_matB.data = NULL;
	
	OPS_Destroy (&ccs_ops);
#if TEST_PAS
	OPS_Destroy (&pas_ops);
#endif
	
#if USE_MPI
	MPI_Finalize();
#endif
	return 0;
}

