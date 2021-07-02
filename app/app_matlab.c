/**
 *    @file  app_matlab.c
 *   @brief  app of matlab
 *
 *  不支持单向量操作 
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/9/13
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include	<assert.h>
#include  	<math.h>
#include   	<memory.h>
#include	<float.h>

#include	"ops.h"
#include    "ops_eig_sol_gcg.h"
#include	"app_lapack.h"
#include	"app_ccs.h"

#if OPS_USE_MATLAB
#include	"mex.h"
#include	"matrix.h"

#define DEBUG 0
/*
 * plhs:
 *   eval, evec, nevConv
 * prhs:
 *   A, B, nev, multiMax, gapMin, block_size, [abs_tol, rel_tol], numIterMax
 *
 * TODO: use options to set other parameters
 */
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	printf("gcge for matlab : https://github.com/Matrials-Of-Numerical-Algebra/GCGE/\n");
	/* 创建 OPS */
	OPS *ccs_ops = NULL;
	OPS_Create (&ccs_ops);
	OPS_CCS_Set (ccs_ops);
	OPS_Setup   (ccs_ops);
	CCSMAT A, *B;
	if (nrhs==0) {
		printf("-------------------------------\n");
		printf("Input Args (at least 2):\n");
		//printf("A, B, nev, abs_tol, rel_tol, nevMax, blockSize, nevInit, numIterMax, gapMin\n");
		printf("A\t\t: Ax = k Bx\tshould be sparse matrix\n");
		printf("B\t\t: Ax = k Bx\tshould be sparse matrix or []\n");
		printf("nev\t\t: The number of requested eigenpairs\n");
		printf("abs_tol\t\t: Absolute tolerance\n");
		printf("rel_tol\t\t: Relative tolerance\n");
		printf("nevMax\t: Maximum number of computed eigenpairs\n");
		printf("blockSize\t: The size of P and W\n");
		printf("nevInit\t\t: The size of working X\n");
		printf("numIterMax\t: Maximum number of iterations\n");
		printf("gapMin\t\t: The minimum relative gap of eigenvalues\n");
		printf("-------------------------------\n");
		printf("Output Args\n");
		printf("eval\t\t: eigenvalues\n");
		printf("evec\t\t: eigenvectors\n");
		printf("nevConv\t: The number of converged eigenpairs\n");
		printf("-------------------------------\n");
		printf("Bug report: liyu@tjufe.edu.cn\n");
		return;		
	}
	if (nrhs < 2) {
		printf("Please input two sparse matrixes A and B\n");
		return;
	}
	/* 开辟GCG工作空间 */
	if (0==mxIsSparse(prhs[0])) {
		printf("A should be a sparse matrix\n");
		return;
	}
	else {
		A.i_row = mxGetIr(prhs[0]); 
		A.j_col = mxGetJc(prhs[0]); 
		A.data  = mxGetPr(prhs[0]); 
		A.nrows = mxGetN (prhs[0]);
		A.ncols = A.nrows;
		printf("A nrows = %d, ncols = %d\n", A.nrows, A.ncols);
	}
	if (0==mxIsSparse(prhs[1])) {
		B = NULL;
		printf("B is a identity matrix\n");
	}
	else {
		B = malloc(sizeof(CCSMAT));
		B->i_row = mxGetIr(prhs[1]); 
		B->j_col = mxGetJc(prhs[1]); 
		B->data  = mxGetPr(prhs[1]); 
		B->nrows = mxGetN(prhs[1]);
		B->ncols = B->nrows;
		printf("B nrows = %d, ncols = %d\n", B->nrows, B->ncols);
	}
	int nevConv, nevMax, nevInit, block_size, numIterMax;
	double tol[2], gapMin;
	
	nevConv = 1;
	if (nrhs >= 3) nevConv = mxGetScalar(prhs[2]);
	if (nevConv >= A.nrows) nevConv = A.nrows-1;
	
	tol[0] = 1e-2;
	if (nrhs >= 4) tol[0] = mxGetScalar(prhs[3]);
	tol[1] = 1e-8;
	if (nrhs >= 5) tol[1] = mxGetScalar(prhs[4]);
	
	nevMax = 2*nevConv; 
	if (nrhs >= 6) nevMax = mxGetScalar(prhs[5]);
	if (nevMax > A.nrows) nevMax = A.nrows;
	
	block_size = nevConv<30?(nevMax-nevConv):nevConv/5;
	if (nrhs >= 7) block_size = mxGetScalar(prhs[6]);
	block_size = block_size<nevMax?block_size:nevMax;
	
	nevInit = nevMax;
	if (nrhs >= 8) nevInit = mxGetScalar(prhs[7]);
	nevInit = nevInit<nevMax?nevInit:nevMax;
	if (nevInit<3*block_size) nevInit = nevMax;
	
	numIterMax = 500;
	if (nrhs >= 9) numIterMax = mxGetScalar(prhs[8]); 
	
	gapMin = 1e-3;
	if (nrhs >= 10) gapMin = mxGetScalar(prhs[9]);
	
	printf("nev = %d, tol[2] = {%e, %e}\n",
			nevConv, tol[0], tol[1]);
	printf("nevMax = %d, blockSize = %d, nevInit = %d\n", 
			nevMax, block_size, nevInit);
	printf("numIterMax = %d, gapMin = %e\n", 
			numIterMax, gapMin);

	void **mv_ws[4]; double *dbl_ws = NULL; int *int_ws = NULL;


	EigenSolverCreateWorkspace_GCG(nevInit,nevMax,block_size,(void *)(&A),
			mv_ws,&dbl_ws,&int_ws,ccs_ops);
	/* GCG设定 */
	EigenSolverSetup_GCG(-1,gapMin,nevInit,nevMax,block_size,
		tol,numIterMax,0,mv_ws,dbl_ws,int_ws,ccs_ops);
	mwSize m = A.nrows, n = nevMax;
	plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(m, n, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);

	
	//double *eval = mxGetDoubles(plhs[0]);
	double *eval = mxGetPr(plhs[0]);
	LAPACKVEC evec;
	evec.nrows = A.nrows; evec.ncols = nevMax; evec.ldd = A.nrows; 
	//evec.data  = mxGetDoubles(plhs[1]);
	evec.data  = mxGetPr(plhs[1]);


#if 1
	/* GCG 参数设定 */
	int    check_conv_max_num    = 50;
	char   initX_orth_method[8]  = "mgs"; 
	int    initX_orth_block_size = -1  ; 
	int    initX_orth_max_reorth = 2   ; double initX_orth_zero_tol   = 2*DBL_EPSILON;
	char   compP_orth_method[8]  = "mgs"; 
	int    compP_orth_block_size = -1  ; 
	int    compP_orth_max_reorth = 2   ; double compP_orth_zero_tol   = 2*DBL_EPSILON;
	char   compW_orth_method[8]  = "mgs";
	int    compW_orth_block_size = -1  ; 	
	int    compW_orth_max_reorth = 2   ; double compW_orth_zero_tol   = 2*DBL_EPSILON;
	int    compW_bpcg_max_iter   = 30  ; double compW_bpcg_rate       = 1e-2; 
	double compW_bpcg_tol        = 1e-12;char   compW_bpcg_tol_type[8]= "abs";
	int    compRR_min_num        = -1  ; double compRR_min_gap        = gapMin; 
	double compRR_tol            = 2*DBL_EPSILON;
#else
	int    check_conv_max_num    = 20;
	char   initX_orth_method[8]   = "mgs"; 
	int    initX_orth_block_size = -1  ; 
	int    initX_orth_max_reorth = 2   ; double initX_orth_zero_tol   = 1e-14;
	char   compP_orth_method[8]   = "mgs"; 
	int    compP_orth_block_size = -1  ; 
	int    compP_orth_max_reorth = 2   ; double compP_orth_zero_tol   = 1e-14;
	char   compW_orth_method[8]   = "mgs";
	int    compW_orth_block_size = -1  ; 	
	int    compW_orth_max_reorth = 2   ; double compW_orth_zero_tol   = 1e-14;
	int    compW_bpcg_max_iter   = 30  ; double compW_bpcg_rate       = 1e-2; 
	double compW_bpcg_tol        = 5e-9; char   compW_bpcg_tol_type[8]= "user";
	int    compRR_min_num        = -1  ; double compRR_min_gap        = gapMin; 
	double compRR_tol            = 1e-16;
#endif

			
	EigenSolverSetParameters_GCG(
			check_conv_max_num   ,
			initX_orth_method    , initX_orth_block_size, 
			initX_orth_max_reorth, initX_orth_zero_tol,
			compP_orth_method    , compP_orth_block_size, 
			compP_orth_max_reorth, compP_orth_zero_tol,
			compW_orth_method    , compW_orth_block_size, 
			compW_orth_max_reorth, compW_orth_zero_tol,
			compW_bpcg_max_iter  , compW_bpcg_rate, 
			compW_bpcg_tol       , compW_bpcg_tol_type, 0,
			compRR_min_num       , compRR_min_gap,
			compRR_tol           ,  
			ccs_ops);		
#if 1	
	/* GCG求解 */
	ccs_ops->EigenSolver((void *)(&A),(void *)B,eval,(void **)(&evec),0,&nevConv,ccs_ops);
#endif	
	printf("numIter = %d, nevConv = %d\n", 
			((GCGSolver*)ccs_ops->eigen_solver_workspace)->numIter, nevConv);
	double nev = nevConv;
	//mxSetDoubles(plhs[2], &nevConv);
	mxSetPr(plhs[2], &nev);
	/* GCG 销毁 */
	EigenSolverDestroyWorkspace_GCG(nevInit,nevMax,block_size,(void *)(&A),
			mv_ws,&dbl_ws,&int_ws,ccs_ops);

	OPS_Destroy (&ccs_ops);
	return;
}

#endif
