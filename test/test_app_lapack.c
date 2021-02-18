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
int TestEigenSolver      (void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);
int TestMultiGrid        (void *A, void *B, struct OPS_ *ops);

#define TEST_PAS 0

int TestAppLAPACK(int argc, char *argv[]) 
{
#if USE_MPI
   MPI_Init(&argc, &argv);
#endif
	
	OPS *lapack_ops = NULL;
	OPS_Create (&lapack_ops);
	OPS_LAPACK_Set (lapack_ops);
	OPS_Setup (lapack_ops);
#if TEST_PAS	
	OPS *pas_ops = NULL;
	OPS_Create (&pas_ops);
	OPS_PAS_Set (pas_ops,lapack_ops);
	OPS_Setup (pas_ops);
#endif	
	void *matA, *matB; OPS *ops;

	int n = 29, row, col; double h = 1.0/(n+1);
	//int n = 1000+7, row, col; double h = 1.0/(n+1);
	//double diag[16] = {1,1,1,2,2,3,4,5,5,5,5,8,8,9,9,9};
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
			if (row == col) lapack_matB.data[row+n*row] = 1.0*h;
			else lapack_matB.data[col+n*row] = 0.0;
		}
	}
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
	pas_matA.XX    = (void*)&lapack_matA;
	pas_matA.QQ[0] = (void*)&lapack_matA;
	pas_matA.QX[0] = (void**)&lapack_vec;
	pas_matB.level_aux = 0; pas_matB.num_levels = 1;
	pas_matB.QQ = malloc(sizeof(void*));
	pas_matB.QX = malloc(sizeof(void**)); 
	pas_matB.XX    = (void*)&lapack_matB;
	pas_matB.QQ[0] = (void*)&lapack_matB;
	pas_matB.QX[0] = (void**)&lapack_vec;
#endif

#if TEST_PAS
	ops = pas_ops; matA = &pas_matA; matB = &pas_matB;
#else
	ops = lapack_ops; matA = (void*)(&lapack_matA); matB = (void*)(&lapack_matB);
#endif 
	
	int para_int = 0; 
	ops->GetOptionFromCommandLine ("-para_int",'i',&para_int,argc,argv, ops);
	ops->Printf("para_int = %d\n",para_int);
	double para_flt = 0.0; 
	ops->GetOptionFromCommandLine ("-para_flt",'f',&para_flt,argc,argv, ops);
	ops->Printf("para_flt = %f\n",para_flt);
	char para_str[128]="test command line"; 
	ops->GetOptionFromCommandLine ("-para_str",'s',para_str,argc,argv, ops);
	ops->Printf("para_str = %s\n",para_str);
	
	
	//
	//TestMultiVec(matA,ops);
	//TestMultiLinearSolver(matA,ops);
	//TestOrth(matA,ops);
	/* The following three fucntions can not be test for PASMAT */
	//TestLinearSolver(matA,ops);
	TestEigenSolver(matA,matB,0,argc,argv,ops);
	//TestMultiGrid(matA,matB,ops);
#if TEST_PAS
 	free(pas_matA.QQ); pas_matA.QQ = NULL;
 	free(pas_matA.QX); pas_matA.QX = NULL;
 	free(pas_matB.QQ); pas_matB.QQ = NULL;
 	free(pas_matB.QX); pas_matB.QX = NULL;
 	free(lapack_vec.data) ; lapack_vec.data  = NULL;
#endif	
	free(lapack_matA.data); lapack_matA.data = NULL;
	free(lapack_matB.data); lapack_matB.data = NULL;
	
	OPS_Destroy (&lapack_ops);
#if TEST_PAS
	OPS_Destroy (&pas_ops);
#endif
	
	double array[6] = {1,2,4,8,16,32};
	int length = 6, ntype = 4; double min_gap = 0.2;
	int min_num = 2;
	int displs[7]; double dbl_ws[6]; int int_ws[6];
	int k, j;
	for (k = 0; k < length; ++k) {
		printf("%f\t",array[k]);
	}
	printf("\n");
	SplitDoubleArray(array, length, 
		ntype, min_gap, min_num, displs, 
		dbl_ws, int_ws);
	for (k = 0; k < ntype; ++k) {
		printf("[%d] %d <= j < %d: ",k,displs[k],displs[k+1]);
		for (j = displs[k]; j < displs[k+1]; ++j) {
			printf("%f\t",array[j]);
		}
		printf("\n");
	}

#if USE_MPI
   MPI_Finalize();
#endif
	return 0;
}

