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
int TestEigenSolverPAS   (void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);
int TestMultiGrid        (void *A, void *B, struct OPS_ *ops);

int TestAppLAPACK(int argc, char *argv[]) 
{
#if OPS_USE_MPI
   MPI_Init(&argc, &argv);
#endif
	
	OPS *lapack_ops = NULL;
	OPS_Create (&lapack_ops);
	OPS_LAPACK_Set (lapack_ops);
	OPS_Setup (lapack_ops);

	void *matA, *matB; OPS *ops;

	//int n = 29, row, col; double h = 1.0/(n+1);
	int n = 800+7, row, col; double h = 1.0/(n+1);
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

	ops = lapack_ops; matA = (void*)(&lapack_matA); matB = (void*)(&lapack_matB);

	int para_int = 0; 
	ops->GetOptionFromCommandLine ("-para_int",'i',&para_int,argc,argv, ops);
	ops->Printf("para_int = %d\n",para_int);
	double para_flt = 0.0; 
	ops->GetOptionFromCommandLine ("-para_flt",'f',&para_flt,argc,argv, ops);
	ops->Printf("para_flt = %f\n",para_flt);
	char para_str[128]="test command line"; 
	ops->GetOptionFromCommandLine ("-para_str",'s',para_str,argc,argv, ops);
	ops->Printf("para_str = %s\n",para_str);
	
	
	//TestVec(matA,ops);
	//TestMultiVec(matA,ops);
	//TestMultiLinearSolver(matA,ops);
	//TestOrth(matA,ops);
	//TestLinearSolver(matA,ops);
	//TestEigenSolverGCG(matA,matB,0,argc,argv,ops);
	TestEigenSolverPAS(matA,matB,0,argc,argv,ops);
	//TestMultiGrid(matA,matB,ops);
	
	free(lapack_matA.data); lapack_matA.data = NULL;
	free(lapack_matB.data); lapack_matB.data = NULL;
	
	OPS_Destroy (&lapack_ops);
	
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

#if OPS_USE_MPI
   MPI_Finalize();
#endif
	return 0;
}

