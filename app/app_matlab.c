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

#if USE_MATLAB
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
	printf("gcge for matlab\n");
	/* 创建 OPS */
	OPS *ccs_ops = NULL;
	OPS_Create (&ccs_ops);
	OPS_CCS_Set (ccs_ops);
	OPS_Setup   (ccs_ops);
	CCSMAT A, *B;

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

	int    nev        = mxGetScalar(prhs[2]); 
	int    multiMax   = mxGetScalar(prhs[3]); 
	double gapMin     = mxGetScalar(prhs[4]);
	int    block_size = mxGetScalar(prhs[5]); 
	double *tol       = mxGetPr(prhs[6]); 
	//double *tol       = mxGetDoubles(prhs[6]); 
	int    numIterMax = mxGetScalar(prhs[7]);
	
	if (nev+multiMax > A.nrows) {
		nev = A.nrows-1;
		multiMax = 1;
	}
	if (block_size <=0 ) {
		block_size = nev/5;
		block_size = (block_size<=0)?nev:block_size;
	}
	else if (block_size>nev){
		block_size = nev;
	}
	
	printf("nev = %d, multiMax = %d, gapMin = %e\n",
			nev, multiMax, gapMin);
	printf("block_size = %d, tol[2] = {%e, %e}, numIterMax = %d\n", 
			block_size, tol[0], tol[1], numIterMax);
	


	void **mv_ws[4]; double *dbl_ws = NULL; int *int_ws = NULL;
	
	int sizeX = nev+multiMax;
	EigenSolverCreateWorkspace_GCG(sizeX,block_size,(void *)(&A),
			mv_ws,&dbl_ws,&int_ws,ccs_ops);
	/* GCG设定 */
	EigenSolverSetup_GCG(nev,multiMax,gapMin,block_size,
		tol,numIterMax,0,mv_ws,dbl_ws,int_ws,ccs_ops);
	mwSize m = A.nrows, n = sizeX;
	plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(m, n, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
	//double *eval = mxGetDoubles(plhs[0]);
	double *eval = mxGetPr(plhs[0]);
	LAPACKVEC evec;
	evec.nrows = A.nrows; evec.ncols = sizeX; evec.ldd = A.nrows; 
	//evec.data  = mxGetDoubles(plhs[1]);
	evec.data  = mxGetPr(plhs[1]);


#if 1
	/* GCG 参数设定 */
	int    check_conv_max_num    = 600;
	const char *initX_orth_method = "mgs"; 
	int    initX_orth_block_size = -1  ; 
	int    initX_orth_max_reorth = 3   ; double initX_orth_zero_tol   = DBL_EPSILON;
	const char *compP_orth_method = "mgs"; 
	int    compP_orth_block_size = -1  ; 
	int    compP_orth_max_reorth = 3   ; double compP_orth_zero_tol   = DBL_EPSILON;
	const char *compW_orth_method = "mgs";
	int    compW_orth_block_size = -1  ; 	
	int    compW_orth_max_reorth = 3   ; double compW_orth_zero_tol   = DBL_EPSILON;
	int    compW_bpcg_max_iter   = 10  ; double compW_bpcg_rate       = 1e-6; 
	double compW_bpcg_tol        = 1e-26; const char *compW_bpcg_tol_type = "abs";
	int    compRR_min_num        = -1  ; double compRR_min_gap        = gapMin; 
	double compRR_tol            = DBL_EPSILON;
#else
	int    check_conv_max_num    = 800;
	const char *initX_orth_method = "mgs"; 
	int    initX_orth_block_size = -1  ; 
	int    initX_orth_max_reorth = 2   ; double initX_orth_zero_tol   = 1e-14;
	const char *compP_orth_method = "mgs"; 
	int    compP_orth_block_size = -1  ; 
	int    compP_orth_max_reorth = 2   ; double compP_orth_zero_tol   = 1e-14;
	const char *compW_orth_method = "mgs";
	int    compW_orth_block_size = -1  ; 	
	int    compW_orth_max_reorth = 2   ; double compW_orth_zero_tol   = 1e-14;
	int    compW_bpcg_max_iter   = 30  ; double compW_bpcg_rate       = 1e-2; 
	double compW_bpcg_tol        = 5e-9; const char *compW_bpcg_tol_type = "user";
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
			compW_bpcg_tol       , compW_bpcg_tol_type,
			compRR_min_num       , compRR_min_gap,
			compRR_tol           ,  
			ccs_ops);		
	
	/* GCG求解 */
	ccs_ops->EigenSolver((void *)(&A),(void *)B,eval,(void **)(&evec),0,&nev,ccs_ops);
	printf("numIter = %d\n", 
			((GCGSolver*)ccs_ops->eigen_solver_workspace)->numIter);
	double nevConv = nev;
	//mxSetDoubles(plhs[2], &nevConv);
	mxSetPr(plhs[2], &nevConv);
	/* GCG 销毁 */
	EigenSolverDestroyWorkspace_GCG(sizeX,block_size,(void *)(&A),
			mv_ws,&dbl_ws,&int_ws,ccs_ops);		
	OPS_Destroy (&ccs_ops);
	return;
}

#endif
