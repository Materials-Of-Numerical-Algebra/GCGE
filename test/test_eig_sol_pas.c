/**
 *    @file  test_eig_sol.c
 *   @brief  特征值求解器测试
 *
 *  PASE and GCGE
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/14
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include	<float.h>
#include    <memory.h>
#include    <time.h>

#include    "ops.h"
#include    "ops_eig_sol_gcg.h"
#include    "ops_eig_sol_pas.h"
#include    "app_lapack.h"


#define DEBUG   0
#define USE_PAS 0
#define USE_AMG 0

int TestEigenSolver(void *A, void *B, int argc, char *argv[], struct OPS_ *ops) 
{
	double *eval, *dbl_ws; int *int_ws;
	void   **evec;
#if DEBUG 
	int    nevMax = 1, multiMax = 2, block_size = 2;
	//int    nevMax = 100, multiMax = 30, block_size = 30;
#else
	//int    nevMax = 10, multiMax = 5, block_size = 3;
	//int    nevMax = 1200, multiMax = 120, block_size = 240;
	int    nevMax = 10000, multiMax = 400, block_size = 200;
	//int    nevMax = 600, multiMax = 120, block_size = 120;
	//int    nevMax = 300, multiMax = 120, block_size = 60;
	//int    nevMax = 150, multiMax = 120, block_size = 30;
	//int    nevMax = 75, multiMax = 75, block_size = 15;
	//int    nevMax = 100, multiMax = 100, block_size = 50;
	//int    nevMax = 100, multiMax = 16, block_size = 16;
#endif
	double gapMin = 1e-5;
	int    nevGiven, nevConv0, nevConv, sizeX, sizeV, idx; 
	//sizeX = nevMax + multiMax;
	sizeX = 2*block_size + multiMax;
	sizeV = sizeX + 2*block_size;
#if USE_PAS
	int length_dbl_ws = 5*sizeX*sizeX+sizeX*sizeV+2*sizeV*sizeV+sizeV*sizeV/2+11*sizeV;
#else
#if DEBUG 
	int length_dbl_ws = 2*sizeV*sizeV+11*sizeV+2*sizeV*sizeV;
#else
	int length_dbl_ws = 2*sizeV*sizeV+10*sizeV+(nevMax+multiMax+2*block_size)+(nevMax+multiMax)*block_size;
	ops->Printf ( "length_dbl_ws = %d\n", length_dbl_ws );
#endif
#endif
	int length_int_ws = 6*sizeV+2*(block_size+2);
	dbl_ws = malloc(length_dbl_ws*sizeof(double));
	int_ws = malloc(length_int_ws*sizeof(double));
	memset(dbl_ws,0,length_dbl_ws*sizeof(double));
	memset(int_ws,0,length_int_ws*sizeof(int));
	srand(0);

	sizeX = nevMax + multiMax;
	sizeV = sizeX + 2*block_size;
#if USE_PAS
	int    max_iter_pas = 2, max_iter_rr = 2, block_size_rr = block_size;
	double tol_pas[2] = {1e-1,1e-1}, tol_rr[3] = {1e-1,1e-1};
#endif
	int    nevInc, max_iter_gcg = 10000;
	double tol_gcg[2] = {1e-12,1e-1};
	//double tol_gcg[2] = {1e-4,1e-1};

	//ops->Printf("mat A:\n");
	//ops->MatView(A,ops);
	//if (B!=NULL) {
	//	ops->Printf("mat B:\n");
	//	ops->MatView(B,ops);
	//}
	eval   = malloc((nevMax+multiMax)*sizeof(double));
	memset(eval  ,0,(nevMax+multiMax)*sizeof(double));
	ops->MultiVecCreateByMat(&evec,(nevMax+multiMax),A,ops);

	double time_start, time_interval;
#if USE_MPI
	time_start = MPI_Wtime();
#else
	time_start = clock();
#endif

#if USE_AMG
	/* AMG 做为线性解法器的 GCG */
	void   **A_array, **B_array, **P_array; 
	int    level, num_levels = 3; 
	ops->MultiGridCreate (&A_array,&B_array,&P_array,&num_levels,A,B,ops);
	--num_levels;	
	//int    amg_max_iter[8] = {2,10,10,20,20,40,40,60};
	int    amg_max_iter[12] = {1,5,5,4,4,4,4,4,4,4,4,4};
	   //10,10,20,20,30,30,40,40,50,50,60};
	/* tol[num_levels] 表示 V cycle 的停止准则, 绝对残量 */ 
	double amg_rate[6] = {1e-2,1e-16,1e-16,1e-16,1e-16,1e-16};
	double amg_tol[6]  = {1e-8,1e-16,1e-16,1e-16,1e-16,1e-16}; 
	void   ***amg_mv_ws[7];
	double *amg_dbl_ws = dbl_ws+length_dbl_ws-6*sizeX;
	int    *amg_int_ws = int_ws;
	/* 非调试情形, amg_mv_ws : 0 1 2 3 4 5 sizeX, 6 sizeV */	
	for (idx = 0; idx < 7; ++idx) {
		amg_mv_ws[idx] = malloc(num_levels*sizeof(void**));
		for (level = 0; level < num_levels; ++level) {
			if (idx < 6) {
#if DEBUG
			   	ops->MultiVecCreateByMat(&(amg_mv_ws[idx][level]),
				 	sizeV,A_array[level],ops);				
			   	ops->MultiVecSetRandomValue(amg_mv_ws[idx][level],0,sizeV,ops);
#else
			   	ops->MultiVecCreateByMat(&(amg_mv_ws[idx][level]),
			   	 	sizeX,A_array[level],ops);				
			   	ops->MultiVecSetRandomValue(amg_mv_ws[idx][level],0,
				   	sizeX,ops);
#endif
			}
			else {
			   ops->MultiVecCreateByMat(&(amg_mv_ws[idx][level]),
				 sizeV,A_array[level],ops);				
			   ops->MultiVecSetRandomValue(amg_mv_ws[idx][level],0,sizeV,ops);
			}
		}
	}

#endif
	
#if USE_PAS
	ops->Printf("===============================================\n");
	ops->Printf("PAS Eigen Solver as preconditioner\n");
	//EigenSolverSetup_PAS(sizeX-3*multiMax/4,3*multiMax/4,gapMin,nevMax,tol_pas,max_iter_pas,
	EigenSolverSetup_PAS(nevMax,multiMax,gapMin,
		sizeX,tol_pas,max_iter_pas,	
		block_size_rr,tol_rr,max_iter_rr,
		A_array,B_array,P_array,num_levels,
		amg_mv_ws,dbl_ws,int_ws,ops);
	nevGiven = 0; nevConv = nevMax;
	//nevGiven = 0; nevConv = sizeX-3*multiMax/4;
	ops->EigenSolver(A,B,eval,evec,nevGiven,&nevConv,ops);

#if USE_MPI
	time_interval = MPI_Wtime() - time_start;
	ops->Printf("Time is %.3f\n", time_interval);
	time_start    = MPI_Wtime();
#else
        time_interval = clock()-time_start;
	ops->Printf("Time is %.3f\n", (double)(time_interval)/CLOCKS_PER_SEC);
	time_start    = clock();
#endif

#endif

	void **gcg_mv_ws[4];
		
#if USE_AMG
	/* TODO: 工作空间如何排布, 是个问题 */
	MultiLinearSolverSetup_BlockAMG(amg_max_iter, amg_rate, amg_tol,
		"abs", A_array, P_array, num_levels, 
		amg_mv_ws, amg_dbl_ws, amg_int_ws, NULL, ops);
	gcg_mv_ws[0] = amg_mv_ws[6][0]; /* sizeV */
	gcg_mv_ws[1] = amg_mv_ws[3][0]; /* block_size */
	gcg_mv_ws[2] = amg_mv_ws[4][0]; /* block_size */
	gcg_mv_ws[3] = amg_mv_ws[5][0]; /* block_size */
#else
	EigenSolverCreateWorkspace_GCG(sizeX,block_size,A,gcg_mv_ws,NULL,NULL,ops);	
#endif	
	//nevInit = 25;
	ops->Printf("===============================================\n");
	ops->Printf("GCG Eigen Solver\n");
	EigenSolverSetup_GCG(nevMax,multiMax,gapMin,block_size,
		tol_gcg,max_iter_gcg,USE_AMG,gcg_mv_ws,dbl_ws,int_ws,ops);
	
	int    check_conv_max_num    = 600;
		
	const char *initX_orth_method = "bgs"; 
	int    initX_orth_block_size = 200 ; 
	int    initX_orth_max_reorth = 3   ; double initX_orth_zero_tol   = DBL_EPSILON;
	//int    initX_orth_max_reorth = 2   ; double initX_orth_zero_tol   = 1e-14;
	
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
			ops);		
		
		
#if USE_PAS
	nevGiven = sizeX; 
#else
	nevGiven = 0; 
#endif
	nevInc = nevMax; nevConv = 0; nevConv0 = 0;

	/* nevInc每次迭代增长的特征值个数需大于最大重数 */
	nevInc = nevInc>multiMax?nevInc:multiMax; 
	EigenSolverSetParametersFromCommandLine_GCG(argc,argv,ops);
	do {
		/* 用户希望收敛的特征对个数 */ 
		nevConv0 = nevMax<(nevConv+nevInc)?nevMax:(nevConv+nevInc);
		nevConv  = nevConv0;
		ops->Printf("nevGiven = %d, nevConv = %d\n",nevGiven,nevConv);
		ops->EigenSolver(A,B,eval,evec,nevGiven,&nevConv,ops);
		ops->Printf("numIter = %d\n",((GCGSolver*)ops->eigen_solver_workspace)->numIter);
		ops->Printf("++++++++++++++++++++++++++++++++++++++++++++++\n");
		nevGiven = nevConv0+multiMax;
	//} while (nevConv < nevMax);
	} while (0);


#if USE_MPI
    time_interval = MPI_Wtime() - time_start;
	ops->Printf("Time is %.3f\n", time_interval);
#else
    time_interval = clock()-time_start;
	ops->Printf("Time is %.3f\n", (double)(time_interval)/CLOCKS_PER_SEC);
#endif
	

#if USE_AMG
	for (idx = 0; idx < 7; ++idx) {		
		for (level = 0; level < num_levels; ++level) {
			ops->MultiVecDestroy(&(amg_mv_ws[idx][level]),sizeV,ops);
		}
		free(amg_mv_ws[idx]);
	}
	ops->Printf("MultiGridDestroy\n");
	++num_levels;
	ops->MultiGridDestroy(&A_array,&B_array,&P_array,&num_levels,ops);
#else	
	EigenSolverDestroyWorkspace_GCG(sizeX,block_size,A,gcg_mv_ws,NULL,NULL,ops);		
#endif

	/* eigenvalues */
	ops->Printf("eigenvalues\n");
	for (idx = 0; idx < nevConv; ++idx) {
		ops->Printf("%6.14e\n",eval[idx]);
	}
	ops->Printf("eigenvectors\n");
	//ops->MultiVecView(evec,0,nevConv,ops);

	ops->MultiVecDestroy(&(evec),sizeX,ops);
	free(eval); free(dbl_ws); free(int_ws);
	return 0;
}
