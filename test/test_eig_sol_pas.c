/**
 *    @file  test_eig_sol.c
 *   @brief  ����ֵ���������
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
#define OPS_USE_AMG 0
#define OPS_USE_PAS 1

int TestEigenSolverPAS(void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops) 
{
	int nevConv = 30, multiMax = 1; double gapMin = 1e-5;
	int nevGiven = 0, block_size = nevConv/5, nevMax = 2*nevConv;
	int nevInit  = nevMax;
	int    max_iter_pas = 50, max_iter_rr = 100, block_size_rr = block_size;
	double tol_pas[2] = {1e-1,1e-8}, tol_rr[3] = {1e-1,1e-8};
	int    max_iter_gcg = 10000;
	double tol_gcg[2] = {1e-1,1e-8};

	double *eval; void   **evec;
	eval = malloc(nevMax*sizeof(double));
	memset(eval,0,nevMax*sizeof(double));
	ops->MultiVecCreateByMat(&evec,nevMax,A,ops);

	double *dbl_ws; int *int_ws; 
	int sizeX = nevMax, sizeV = sizeX + 2*block_size;
	/* �����ռ� Ӧ�������Ľ� δ���� shift ���� */
	int length_dbl_ws = 5*sizeX*sizeX+sizeX*sizeV
		+2*sizeV*sizeV+11*sizeV+sizeX*block_size_rr;
	int length_int_ws = 6*sizeV+2*(block_size_rr+3);
	dbl_ws = malloc(length_dbl_ws*sizeof(double));
	memset(dbl_ws,0,length_dbl_ws*sizeof(double));
	int_ws = malloc(length_int_ws*sizeof(int));
	memset(int_ws,0,length_int_ws*sizeof(int));
	srand(0);

	int idx; 
	/* AMG ��Ϊ���Խⷨ���� GCG */
	void   **A_array, **B_array, **P_array; 
	int    level, num_levels = 3;
	ops->Printf("TestEigenSolverPAS\n");
	ops->MultiGridCreate (&A_array,&B_array,&P_array,&num_levels,A,B,ops);
	--num_levels;	
	ops->Printf("TestEigenSolverPAS\n");
	void   ***amg_mv_ws[7];
	/* �ǵ�������, amg_mv_ws : 0 1 2 3 4 5 sizeX, 6 sizeV */	
	for (idx = 0; idx < 7; ++idx) {
		amg_mv_ws[idx] = malloc(num_levels*sizeof(void**));
		for (level = 0; level < num_levels; ++level) {
			if (idx < 6) {
			   	ops->MultiVecCreateByMat(&(amg_mv_ws[idx][level]),
			   	 	sizeX,A_array[level],ops);
			   	ops->MultiVecSetRandomValue(amg_mv_ws[idx][level],0,
				   	sizeX,ops);
			}
			else {
				ops->MultiVecCreateByMat(&(amg_mv_ws[idx][level]),
					sizeV,A_array[level],ops);
				ops->MultiVecSetRandomValue(amg_mv_ws[idx][level],0,
					sizeV,ops);
			}
		}
	}

	ops->Printf("TestEigenSolverPAS\n");
	double time_start, time_interval;
	time_start = ops->GetWtime();

#if OPS_USE_PAS
	ops->Printf("===============================================\n");
	ops->Printf("PAS Eigen Solver as preconditioner\n");
	//EigenSolverSetup_PAS(sizeX-3*multiMax/4,3*multiMax/4,gapMin,nevMax,tol_pas,max_iter_pas,
	EigenSolverSetup_PAS(multiMax,gapMin,nevMax,
		block_size   ,tol_pas,max_iter_pas,	
		block_size_rr,tol_rr ,max_iter_rr ,
		A_array,B_array,P_array,num_levels,
		amg_mv_ws,dbl_ws,int_ws,ops);
	/* nevGiven ��Ϊ��ʱ����, ��¼Ҫ����������Ը��� */
	nevGiven = nevConv;
	ops->EigenSolver(A,B,eval,evec,nevGiven,&nevGiven,ops);
#endif 

	time_interval = ops->GetWtime() - time_start;
	ops->Printf("Time is %.3f\n", time_interval);
	time_start    = ops->GetWtime();

	/* �� PAS �õ���������Ϊ��ֵ ���µ��� GCG */
	void **gcg_mv_ws[4];
	gcg_mv_ws[0] = amg_mv_ws[6][0]; /* sizeV */
	gcg_mv_ws[1] = amg_mv_ws[3][0]; /* block_size */
	gcg_mv_ws[2] = amg_mv_ws[4][0]; /* block_size */
	gcg_mv_ws[3] = amg_mv_ws[5][0]; /* block_size */
		
#if OPS_USE_AMG
	/* amg_(rate/tol)[num_levels] ��ʾ V cycle 
	 * �� num_levels ���� CG �����Ĳ���
	 * amg_max_iter[2*num_levels+1,2*num_levels+2] ��ʾ V cycle 
	 * �� num_levels ����ǰ��⻬����
	 * amg_(max_iter/rate/tol)[0] ��ʾ V cycle 
	 * ����Ĳ��� */
	int    amg_max_iter[12] = {3   , 40,40, 40,40, 40,40, 40,40, 40,40,  40};
	double amg_rate[6]      = {1e-2, 1e-16, 1e-16, 1e-16, 1e-16, 1e-16};
	double amg_tol[6]       = {1e-8, 1e-16, 1e-16, 1e-16, 1e-16, 1e-16}; 
	/* һ���ǴӺ���ȡ, ��Ϊ ǰ�沿�ֻ�������ֵ�������õ�, ����������� */ 
	/* ֻ�� 6*sizeX ��  */
	double *amg_dbl_ws = dbl_ws+length_dbl_ws-(6*sizeX);
	/* ֻ�� 1.5*sizeX+2 */
	int    *amg_int_ws = int_ws+length_int_ws-(sizeX+sizeX/2+2);

	/* TODO: �����ռ�����Ų�, �Ǹ����� */
	MultiLinearSolverSetup_BlockAMG(amg_max_iter, amg_rate, amg_tol,
		"abs", A_array, P_array, num_levels, 
		amg_mv_ws, amg_dbl_ws, amg_int_ws, NULL, ops);
#endif	

	ops->Printf("===============================================\n");
	ops->Printf("GCG Eigen Solver\n");	
		/* �趨 ops �е�����ֵ������� GCG */
	EigenSolverSetup_GCG(multiMax,gapMin,nevInit,nevMax,block_size,
		tol_gcg,max_iter_gcg,OPS_USE_AMG,gcg_mv_ws,dbl_ws,int_ws,ops);
	
	/* չʾ�㷨���в��� */
	int    check_conv_max_num    = 50   ;
		
	char   initX_orth_method[8]  = "mgs"; 
	int    initX_orth_block_size = -1   ; 
	int    initX_orth_max_reorth = 2    ; double initX_orth_zero_tol    = 2*DBL_EPSILON;//1e-12
	
	char   compP_orth_method[8]  = "mgs"; 
	int    compP_orth_block_size = -1   ; 
	int    compP_orth_max_reorth = 2    ; double compP_orth_zero_tol    = 2*DBL_EPSILON;//1e-12
	
	char   compW_orth_method[8]  = "mgs";
	int    compW_orth_block_size = -1   ; 	
	int    compW_orth_max_reorth = 2    ;  double compW_orth_zero_tol   = 2*DBL_EPSILON;//1e-12
	int    compW_bpcg_max_iter   = 30   ;  double compW_bpcg_rate       = 1e-2; 
	double compW_bpcg_tol        = 1e-12;  char   compW_bpcg_tol_type[8] = "abs";
	
	int    compRR_min_num        = -1   ;  double compRR_min_gap        = gapMin;
	double compRR_tol            = 2*DBL_EPSILON;
			
	/* �趨 GCG ���㷨���� */
	EigenSolverSetParameters_GCG(
			check_conv_max_num   ,
			initX_orth_method    , initX_orth_block_size, 
			initX_orth_max_reorth, initX_orth_zero_tol  ,
			compP_orth_method    , compP_orth_block_size, 
			compP_orth_max_reorth, compP_orth_zero_tol  ,
			compW_orth_method    , compW_orth_block_size, 
			compW_orth_max_reorth, compW_orth_zero_tol  ,
			compW_bpcg_max_iter  , compW_bpcg_rate      , 
			compW_bpcg_tol       , compW_bpcg_tol_type  , 0, // with shift 
			compRR_min_num       , compRR_min_gap       ,
			compRR_tol           ,  
			ops);		

	/* �����л�ȡ GCG ���㷨���� ���� �� BUG, 
	 * ��Ӧ�øı� nevMax nevInit block_size, ��Щ�빤���ռ��й� */
#if OPS_USE_PAS
	nevGiven = nevMax;
#else
	nevGiven = 0; 
#endif
	EigenSolverSetParametersFromCommandLine_GCG(argc,argv,ops);
	ops->Printf("nevGiven = %d, nevConv = %d, nevMax = %d, block_size = %d, nevInit = %d\n",
			nevGiven,nevConv,nevMax,block_size,nevInit);
	ops->EigenSolver(A,B,eval,evec,nevGiven,&nevConv,ops);
	ops->Printf("numIter = %d, nevConv = %d\n",
			((GCGSolver*)ops->eigen_solver_workspace)->numIter, nevConv);
	ops->Printf("++++++++++++++++++++++++++++++++++++++++++++++\n");
	
	time_interval = ops->GetWtime() - time_start;
	ops->Printf("Time is %.3f\n", time_interval);
	

	for (idx = 0; idx < 7; ++idx) {		
		for (level = 0; level < num_levels; ++level) {
			if (idx < 6)
				ops->MultiVecDestroy(&(amg_mv_ws[idx][level]),sizeX,ops);
			else
				ops->MultiVecDestroy(&(amg_mv_ws[idx][level]),sizeV,ops);
		}
		free(amg_mv_ws[idx]);
	}
	ops->Printf("MultiGridDestroy\n");
	++num_levels;
	ops->MultiGridDestroy(&A_array,&B_array,&P_array,&num_levels,ops);

	/* eigenvalues */
	ops->Printf("eigenvalues\n");
	for (idx = 0; idx < nevConv; ++idx) {
		ops->Printf("%d: %6.14e\n",idx+1,eval[idx]);
	}
	ops->Printf("eigenvectors\n");
	//ops->MultiVecView(evec,0,nevConv,ops);

	ops->MultiVecDestroy(&(evec),sizeX,ops);
	free(eval); free(dbl_ws); free(int_ws);
	return 0;
}
