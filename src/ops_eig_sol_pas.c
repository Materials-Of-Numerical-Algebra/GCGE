/**
 *    @file  ops_eig_sol_pas.c
 *   @brief  特征值求解器 PAS 
 *
 *  特征值求解器 PAS
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/18
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include	<string.h>
#include    <math.h>
#include    <time.h>
#include    <memory.h>
#include    <assert.h>


#include    "ops_eig_sol_pas.h"


#define     DEBUG 0
#define     TIME_PAS 1

typedef struct TimePAS_ {
	double checkconv_time;
	double compRR_time;
	double compRV_time;
	double compN_time;
    double orthX_time;
	double promX_time;           
    double time_total;
} TimePAS;

struct TimePAS_ time_pas = {0.0,0.0,0.0,0.0,0.0,0.0,0.0};

static int sizeN, startN, endN;
static int sizeC, sizeX;
static int level, level_aux;

static void ***mv_ws[5]; 
static double *dbl_ws; 
static int    *int_ws;
static struct OPS_ *ops_pas;
static struct PASSolver_ *pas_solver;

static void ComputeRayleighRitz(PASMAT *ss_matA, PASMAT *ss_matB,
		double *ss_eval, PASVEC *ss_evec)
{
#if TIME_PAS
#if USE_MPI
    time_pas.compRR_time -= MPI_Wtime();
#else
    time_pas.compRR_time -= clock();
#endif
#endif
	//PASSolver *pas = (PASSolver*)ops_pas->app_ops->eigen_solver_workspace;
	int start[2], end[2], idx, nevGiven, nevConv;
	void **mv_ws_gcg[4] = {NULL};
	int sizeV = sizeX + 2*pas_solver->block_size_rr;
	if (level == level_aux) {
		//ops_pas->app_ops->MultiVecCreateByMat(&(mv_ws_gcg[0]),sizeV,
		//	ss_matA->QQ[level],ops_pas->app_ops);
		mv_ws_gcg[0] = mv_ws[4][level]; /* sizeV */ 
		for (idx = 1; idx < 4; ++idx) {
			mv_ws_gcg[idx] = mv_ws[idx][level];
		}
		void(*eig_sol)(void*,void*,double*,void**,int,int*,struct OPS_*);
		void *ws;
		eig_sol = ops_pas->app_ops->EigenSolver;
		ws      = ops_pas->app_ops->eigen_solver_workspace;
		EigenSolverSetup_GCG(
			pas_solver->nevMax,pas_solver->multiMax,pas_solver->gapMin,
			pas_solver->block_size_rr,pas_solver->tol_rr,pas_solver->numIterMax_rr,
			0,mv_ws_gcg,dbl_ws,int_ws, 
			ops_pas->app_ops);
		/* GCG 算法内部参数 */
		EigenSolverSetParameters_GCG(
			pas_solver->compRR_gcg_check_conv_max_num,
			pas_solver->compRR_gcg_initX_orth_method, 
			pas_solver->compRR_gcg_initX_orth_block_size, 
			pas_solver->compRR_gcg_initX_orth_max_reorth, 
			pas_solver->compRR_gcg_initX_orth_zero_tol,
			pas_solver->compRR_gcg_compP_orth_method, 
			pas_solver->compRR_gcg_compP_orth_block_size, 
			pas_solver->compRR_gcg_compP_orth_max_reorth, 
			pas_solver->compRR_gcg_compP_orth_zero_tol,
			pas_solver->compRR_gcg_compW_orth_method, 
			pas_solver->compRR_gcg_compW_orth_block_size, 
			pas_solver->compRR_gcg_compW_orth_max_reorth, 
			pas_solver->compRR_gcg_compW_orth_zero_tol,
			pas_solver->compRR_gcg_compW_cg_max_iter, 
			pas_solver->compRR_gcg_compW_cg_rate, 
			pas_solver->compRR_gcg_compW_cg_tol, 
			pas_solver->compRR_gcg_compW_cg_tol_type,
			pas_solver->compRR_gcg_compRR_min_num,
			pas_solver->compRR_gcg_compRR_min_gap, 
			pas_solver->compRR_gcg_compRR_tol, 
			ops_pas->app_ops);		
		
#if DEBUG
		ops_pas->Printf("initi ss_evec->q[%d]\n",level);
		ops_pas->app_ops->MultiVecView(ss_evec->q[level],0,sizeX,ops_pas->app_ops);
#endif
		nevGiven = 0; nevConv = pas_solver->nevMax;
		//nevGiven = 0; nevConv = sizeX;
		ops_pas->app_ops->EigenSolver(ss_matA->QQ[level],ss_matB->QQ[level],ss_eval,ss_evec->q[level],
			nevGiven,&nevConv,ops_pas->app_ops);
		ops_pas->Printf("nevConv = %d, pas->nevMax = %d\n", nevConv, pas_solver->nevMax);
		//assert(nevConv>=pas->nevMax);
#if DEBUG
		ops_pas->Printf("final ss_evec->q[%d]\n",level);
		ops_pas->app_ops->MultiVecView(ss_evec->q[level],0,sizeX,ops_pas->app_ops);
#endif
		//ops_pas->app_ops->MultiVecDestroy(&(mv_ws_gcg[0]),sizeV,ops_pas->app_ops);
		for (idx = 0; idx < 4; ++idx) {
			mv_ws_gcg[idx] = NULL;
		}
		ops_pas->app_ops->EigenSolver            = eig_sol;
		ops_pas->app_ops->eigen_solver_workspace = ws     ;		
	}
	else {
		int length, incx, incy; double *source, *destin;
		/* 计算 XAX 部分, 不可以忽略 C 部分 */
		double *xAx = ((LAPACKMAT*)ss_matA->XX)->data;
		start[0] = sizeC; end[0] = sizeX;
		start[1] = sizeC; end[1] = sizeX;
		ops_pas->app_ops->MultiVecQtAP('S','S',
			ss_evec->q[level],ss_matA->QQ[level],ss_evec->q[level],0,
			start,end,xAx+sizeX*sizeC+sizeC,sizeX,
			mv_ws[1][level],ops_pas->app_ops);
			
		start[0] = sizeC; end[0] = sizeX;
		start[1] = 0    ; end[1] = sizeC;
		ops_pas->app_ops->MultiVecQtAP('S','N',
			ss_evec->q[level],ss_matA->QQ[level],ss_evec->q[level],0,
			start,end,xAx+sizeC,sizeX,
			mv_ws[1][level],ops_pas->app_ops);
		/* 对称化 */
		length = sizeX-sizeC;
		source = xAx+sizeC      ; incx = 1    ;
		destin = xAx+sizeX*sizeC; incy = sizeX;					
		for (idx = 0; idx < sizeC; ++idx) {
			dcopy(&length,source,&incx,destin,&incy);
			source += sizeX; destin += 1;
		}			
			
		/* 计算 QAX 部分, 忽略 C 部分 */
		start[0] = sizeC; end[0] = sizeX;
		start[1] = sizeC; end[1] = sizeX;
		ops_pas->app_ops->MatDotMultiVec(
			ss_matA->QQ[level],ss_evec->q[level],ss_matA->QX[level],
			start,end,ops_pas->app_ops);
		ops_pas->app_ops->MultiVecFromItoJ(ss_matA->P,level,level_aux,
			ss_matA->QX[level],ss_matA->QX[level_aux],start,end,
			ss_matA->QX,ops_pas->app_ops);			
#if DEBUG
		//ops_pas->Printf("ss_matA level = %d\n",level);
		//ops_pas->MatView((void*)ss_matA,ops_pas);
		//ops_pas->Printf("ss_matB level = %d\n",level);
		//ops_pas->MatView((void*)ss_matB,ops_pas);
#endif
		/* 调用GCG进行特征值问题的求解 */		
		//ops_pas->MultiVecCreateByMat(&(mv_ws_gcg[0]),sizeV,
		//	(void*)ss_matA,ops_pas);
		PASVEC pas_vec_ws[4]; LAPACKVEC lapack_vec_ws[4];
		for (idx = 0; idx < 4; ++idx) {
			mv_ws_gcg[idx] = (void**)&(pas_vec_ws[idx]);
			pas_vec_ws[idx].level_aux  = level_aux;
			pas_vec_ws[idx].num_levels = ss_matA->num_levels;
			pas_vec_ws[idx].P          = ss_matA->P;
			pas_vec_ws[idx].x          = (void**)&(lapack_vec_ws[idx]);
		}
		pas_vec_ws[0].q = mv_ws[4]; /* sizeV */		
		pas_vec_ws[1].q = mv_ws[1];			
		pas_vec_ws[2].q = mv_ws[2];			
		pas_vec_ws[3].q = mv_ws[3];			

		lapack_vec_ws[0].ldd   = sizeX;
		lapack_vec_ws[0].nrows = sizeX;
		lapack_vec_ws[0].ncols = sizeV;
		lapack_vec_ws[0].data  = dbl_ws;
		for (idx = 1; idx < 4; ++idx) {
			lapack_vec_ws[idx].ldd   = sizeX;
			lapack_vec_ws[idx].nrows = sizeX;
			lapack_vec_ws[idx].ncols = sizeX;
			lapack_vec_ws[idx].data  = dbl_ws+sizeX*sizeV+sizeX*sizeX*(idx-1);
		}
		
		EigenSolverSetup_GCG(//sizeX,0,
			pas_solver->nevMax,pas_solver->multiMax,pas_solver->gapMin,
			pas_solver->block_size_rr,pas_solver->tol_rr,pas_solver->numIterMax_rr,			
			0,mv_ws_gcg,dbl_ws+sizeX*sizeV+3*sizeX*sizeX,int_ws,
			ops_pas);		
		/* GCG 算法内部参数 */
		EigenSolverSetParameters_GCG(
			pas_solver->compRR_gcg_check_conv_max_num,
			pas_solver->compRR_gcg_initX_orth_method, 
			pas_solver->compRR_gcg_initX_orth_block_size, 
			pas_solver->compRR_gcg_initX_orth_max_reorth, 
			pas_solver->compRR_gcg_initX_orth_zero_tol,
			pas_solver->compRR_gcg_compP_orth_method, 
			pas_solver->compRR_gcg_compP_orth_block_size, 
			pas_solver->compRR_gcg_compP_orth_max_reorth, 
			pas_solver->compRR_gcg_compP_orth_zero_tol,
			pas_solver->compRR_gcg_compW_orth_method, 
			pas_solver->compRR_gcg_compW_orth_block_size, 
			pas_solver->compRR_gcg_compW_orth_max_reorth, 
			pas_solver->compRR_gcg_compW_orth_zero_tol,
			pas_solver->compRR_gcg_compW_cg_max_iter, 
			pas_solver->compRR_gcg_compW_cg_rate, 
			pas_solver->compRR_gcg_compW_cg_tol, 
			pas_solver->compRR_gcg_compW_cg_tol_type,
			pas_solver->compRR_gcg_compRR_min_num,
			pas_solver->compRR_gcg_compRR_min_gap, 
			pas_solver->compRR_gcg_compRR_tol, 
			ops_pas);	
#if DEBUG
		ops_pas->Printf("initi ss_evec, level = %d\n",level);
		ops_pas->MultiVecView((void**)ss_evec,0,sizeX,ops_pas);
#endif
		nevGiven = sizeX; nevConv = pas_solver->nevMax;
		//nevGiven = sizeX; nevConv = sizeX;
		ops_pas->EigenSolver((void*)ss_matA,(void*)ss_matB,ss_eval,(void**)ss_evec,
			nevGiven,&nevConv,ops_pas);
		ops_pas->Printf("nevConv = %d, pas_solver->nevMax = %d\n", nevConv, pas_solver->nevMax);
		//assert(nevConv>=pas->nevMax);
		
#if DEBUG
		ops_pas->Printf("before RQ ss_evec, level = %d\n",level);
		ops_pas->MultiVecView((void**)ss_evec,0,sizeX,ops_pas);
#endif
		/* 对特征子空间进行RQ分解, 使得对应的稠密矩阵部分是上三角矩阵 */
		//MakeSubspaceEvecGood(ss_evec,ss_eval);		
		
#if DEBUG
		ops_pas->Printf("final ss_evec, level = %d\n",level);
		ops_pas->MultiVecView((void**)ss_evec,0,sizeX,ops_pas);
#endif	
		//ops_pas->MultiVecDestroy(&(mv_ws_gcg[0]),sizeV,ops_pas);
		for (idx = 0; idx < 4; ++idx) {
			mv_ws_gcg[idx] = NULL;
		}			
	}
#if DEBUG
	ops_pas->Printf("eigenvalues:\n");
	for (idx = 0; idx < nevConv; ++idx) {
		ops_pas->Printf("%6.4e\n",ss_eval[idx]);
	}
#endif
#if TIME_PAS
#if USE_MPI
    time_pas.compRR_time += MPI_Wtime();
#else
    time_pas.compRR_time += clock();
#endif
#endif
	return;
}
static void ComputeRitzVec(void **ritz_vec, PASVEC *ss_evec)
{
#if TIME_PAS
#if USE_MPI
    time_pas.compRV_time -= MPI_Wtime();
#else
    time_pas.compRV_time -= clock();
#endif
#endif
	int start[2], end[2];
	start[0] = 0    ; end[0] = sizeX;
	start[1] = sizeC; end[1] = sizeX;
	//start[1] = 0    ; end[1] = sizeX;
	LAPACKVEC *lapack_vec = (LAPACKVEC*)ss_evec->x;
	double *coef = lapack_vec->data+lapack_vec->ldd*sizeC;
	//double *coef = lapack_vec->data;
	/* 注意: level > 0 时, ritz_vec == ss_evec->q[level] */
	ops_pas->app_ops->MultiVecLinearComb(ss_evec->q[level],mv_ws[0][level],0,
		start,end,coef,lapack_vec->ldd,NULL,0,ops_pas->app_ops);
	start[0] = sizeC; end[0] = sizeX;
	start[1] = sizeC; end[1] = sizeX;
	ops_pas->app_ops->MultiVecFromItoJ(ss_evec->P,level_aux,level,
		ss_evec->q[level_aux],ritz_vec,start,end,mv_ws[1],ops_pas->app_ops);
	ops_pas->app_ops->MultiVecAxpby(1.0,mv_ws[0][level],1.0,ritz_vec,
		start,end,ops_pas->app_ops);
#if DEBUG
	ops_pas->Printf("ritz_vec\n");
	ops_pas->app_ops->MultiVecView(ritz_vec,0,sizeX,ops_pas);
#endif
#if TIME_PAS
#if USE_MPI
    time_pas.compRV_time += MPI_Wtime();
#else
    time_pas.compRV_time += clock();
#endif
#endif
	return;	
}
static int  CheckConvergence(void *A, void *B, double *ss_eval, void **ritz_vec, 
	int numCheck, double *tol)
{
#if TIME_PAS
#if USE_MPI
    time_pas.checkconv_time -= MPI_Wtime();
#else
    time_pas.checkconv_time -= clock();
#endif
#endif
#if DEBUG
	ops_pas->Printf("numCheck = %d\n", numCheck);
#endif
	int start[2], end[2], idx;
	start[0] = startN; end[0] = start[0]+numCheck;
	start[1] = 0     ; end[1] = numCheck;	
	ops_pas->app_ops->MatDotMultiVec(A,ritz_vec,mv_ws[0][level],
		start,end,ops_pas->app_ops);	
	ops_pas->app_ops->MatDotMultiVec(B,ritz_vec,mv_ws[1][level],
		start,end,ops_pas->app_ops);	
	/* lambda Bx */
	ops_pas->app_ops->MultiVecLinearComb(NULL,mv_ws[1][level],0,start,end,
		NULL,0,ss_eval+startN,1,ops_pas->app_ops);
	start[0] = 0     ; end[0] = numCheck;
	start[1] = 0     ; end[1] = numCheck;
	/* Ax - lambda Bx */
	ops_pas->app_ops->MultiVecAxpby(-1.0,mv_ws[1][level],
		1.0,mv_ws[0][level],
		start,end,ops_pas->app_ops);
	ops_pas->app_ops->MultiVecInnerProd('D',mv_ws[0][level],
		mv_ws[0][level],0,start,end,dbl_ws,1,ops_pas->app_ops);
	for (idx = 0; idx < numCheck; ++idx) {
		dbl_ws[idx] = sqrt(dbl_ws[idx]);
		ops_pas->Printf("PAS: [%d] %6.4e (%6.4e)\n",startN+idx,ss_eval[startN+idx],dbl_ws[idx]);
	}
	ops_pas->Printf("tol = %6.4e, %6.4e\n", tol[0], tol[1]);
	for (idx = 0; idx < numCheck; ++idx) {
		if (dbl_ws[idx] > tol[0] || 
				dbl_ws[idx] > ss_eval[startN+idx]*tol[1]) break;
	}
	/* 判断重根 */
	for ( ; idx > 0; --idx) {
		if ( fabs((ss_eval[startN+idx-1]-ss_eval[startN+idx])/ss_eval[startN+idx-1]) > 1e-2 ) {
			break;
		}
	}
#if TIME_PAS
#if USE_MPI
    time_pas.checkconv_time += MPI_Wtime();
#else
    time_pas.checkconv_time += clock();
#endif
#endif
	return (sizeC+idx);
}
static void PromoteX(void ***X, void **P)
{
#if TIME_PAS
#if USE_MPI
    time_pas.promX_time -= MPI_Wtime();
#else
    time_pas.promX_time -= clock();
#endif
#endif
	int start[2], end[2];
	start[0] = 0; end[0] = sizeX;
	start[1] = 0; end[1] = sizeX;
	ops_pas->app_ops->MultiVecFromItoJ(P,level,level-1,
		X[level],X[level-1],start,end,X,ops_pas->app_ops);
#if TIME_PAS
#if USE_MPI
    time_pas.promX_time += MPI_Wtime();
#else
    time_pas.promX_time += clock();
#endif
#endif
	return;
}
static void ComputeN(void **ritz_vec, void *A, void *B, double *ss_eval)
{
#if TIME_PAS
#if USE_MPI
    time_pas.compN_time -= MPI_Wtime();
#else
    time_pas.compN_time -= clock();
#endif
#endif
	int start[2], end[2];
	/* set b */
	void **b = mv_ws[0][level];
	start[0]  = startN; end[0] = endN;
	start[1]  = startN; end[1] = endN;
	/* lambda Bx */
	ops_pas->app_ops->MatDotMultiVec(B,ritz_vec,b,
		start,end,ops_pas->app_ops);
	ops_pas->app_ops->MultiVecLinearComb(NULL,b,0,start,end,
		NULL,0,ss_eval+start[0],1,ops_pas->app_ops);
	/* solve x */
	void ***amg_mv_ws[5] = {
			mv_ws[0]+level,mv_ws[1]+level,mv_ws[2]+level,
			mv_ws[3]+level,mv_ws[4]+level};
	
	if (pas_solver->compN_user_defined_multi_linear_solver==0) {		
	   	ops_pas->Printf("level = %d, num_levels = %d\n", level, pas_solver->num_levels);
	   	MultiLinearSolverSetup_BlockAMG(
	   		pas_solver->compN_bamg_max_iter,
	   		pas_solver->compN_bamg_rate,
	   		pas_solver->compN_bamg_tol,
	   		pas_solver->compN_bamg_tol_type,
			(pas_solver->A)+level, (pas_solver->P)+level, (pas_solver->num_levels)-level,
		 	amg_mv_ws, dbl_ws, int_ws, NULL, ops_pas->app_ops);
	}

#if DEBUG
	ops_pas->Printf("initi N\n");
	ops_pas->app_ops->MultiVecView(ritz_vec,start[1],end[1],ops_pas->app_ops);
#endif
	ops_pas->app_ops->MultiLinearSolver(A,b,ritz_vec,
		start,end,ops_pas->app_ops);
#if DEBUG
	ops_pas->Printf("final N\n");
	ops_pas->app_ops->MultiVecView(ritz_vec,start[1],end[1],ops_pas->app_ops);
#endif
#if TIME_PAS
#if USE_MPI
    time_pas.compN_time += MPI_Wtime();
#else
    time_pas.compN_time += clock();
#endif
#endif
	return;
}
static void OrthXtoQ(PASVEC *ss_evec, void **P, void **B, void **ritz_vec)
{
#if TIME_PAS
#if USE_MPI
    time_pas.orthX_time -= MPI_Wtime();
#else
    time_pas.orthX_time -= clock();
#endif
#endif
	void **X  = ss_evec->q[level];
	void **x  = ss_evec->q[level_aux];
	int start[2], end[2];
	/* set b */
	void ***b = mv_ws[0];
	start[0] = startN; end[0] = sizeX;
	start[1] = startN; end[1] = sizeX;


	//start[0] = 0; end[0] = sizeX;
	//start[1] = 0; end[1] = sizeX;
	if (level == 0) {
		ops_pas->app_ops->MatDotMultiVec(B[level],ritz_vec,b[level],
			start,end,ops_pas->app_ops);
	}
	else {
		ops_pas->app_ops->MatDotMultiVec(B[level],X,b[level],
			start,end,ops_pas->app_ops);
	}
	ops_pas->app_ops->MultiVecFromItoJ(P,level,level_aux,
		b[level],b[level_aux],start,end,b,ops_pas->app_ops);
	/* initialize x (Qt ritz_vec) */
	/* DO NOT NEED to set iniital values */
#if 0
	if (level == 0) {
	   ops_pas->app_ops->MultiVecFromItoJ(P,level,level_aux,
		 ritz_vec,x,start,end,mv_ws[1],ops_pas->app_ops);
	}
	else {
	   ops_pas->app_ops->MultiVecFromItoJ(P,level,level_aux,
		 X,x,start,end,mv_ws[1],ops_pas->app_ops);
	}
#endif
	/* solve x */
	int idx;
	int amg_max_iter[8]; double amg_rate[8], amg_tol[8];
	/* 前光滑 */
	for (idx = 1; idx < 8; idx+=2) {
		amg_max_iter[idx] = 4;
		amg_rate[idx] = 1e-2; amg_tol[idx] = 1e-14;
	}
	/* 后光滑 */ 
	for (idx = 0; idx < 8; idx+=2) {
		amg_max_iter[idx] = 4;
		amg_rate[idx] = 1e-2; amg_tol[idx] = 1e-14;
	}

	if (pas_solver->orthX_user_defined_multi_linear_solver==0) {
		/* 最细层 */ 
		amg_max_iter[0] = pas_solver->orthX_ls_max_iter;
		amg_rate[0] = pas_solver->orthX_ls_rate;
		amg_tol[0]  = pas_solver->orthX_ls_tol;
		void ***amg_mv_ws[5] = {
			mv_ws[0]+level_aux,mv_ws[1]+level_aux,mv_ws[2]+level_aux,
			mv_ws[3]+level_aux,mv_ws[4]+level_aux};
		MultiLinearSolverSetup_BlockAMG(
		   	amg_max_iter,amg_rate,amg_tol,
		   	pas_solver->orthX_ls_tol_type,
			B+level_aux, P+level_aux, (pas_solver->num_levels)-level_aux,
			amg_mv_ws, dbl_ws, int_ws, NULL, ops_pas->app_ops);
	}	
#if DEBUG
	ops_pas->Printf("initi x\n");
	ops_pas->app_ops->MultiVecView(x,start[1],end[1],ops_pas->app_ops);
#endif
	ops_pas->app_ops->MultiLinearSolver(B[level_aux],b[level_aux],
		x,start,end,ops_pas->app_ops);
#if DEBUG
	ops_pas->Printf("final x\n");
	ops_pas->app_ops->MultiVecView(x,start[1],end[1],ops_pas->app_ops);
#endif
	ops_pas->app_ops->MultiVecFromItoJ(P,level_aux,level,
		x,b[level],start,end,mv_ws[1],ops_pas->app_ops);
	if (level == 0) {
		ops_pas->app_ops->MultiVecAxpby(1.0,ritz_vec,0.0,X,
			start,end,ops_pas->app_ops);
		ops_pas->app_ops->MultiVecAxpby(-1.0,b[level],1.0,X,
			start,end,ops_pas->app_ops);		
	}
	else {
		ops_pas->app_ops->MultiVecAxpby(-1.0,b[level],1.0,X,
			start,end,ops_pas->app_ops);
	}
	
#if DEBUG
	ops_pas->Printf("before Orth X - QB^{-1}QtBX\n");
	ops_pas->app_ops->MultiVecView(X, 0, sizeX, ops_pas);
#endif
	/* orth X self */
	int endX = sizeX;
	if (0 == strcmp("mgs", pas_solver->orthX_orth_method))
		MultiVecOrthSetup_ModifiedGramSchmidt(
				pas_solver->orthX_orth_block_size,
				pas_solver->orthX_orth_max_reorth,
				pas_solver->orthX_orth_zero_tol,
				mv_ws[3][level],dbl_ws,ops_pas->app_ops);
	else if (0 == strcmp("bgs", pas_solver->orthX_orth_method))
		MultiVecOrthSetup_BinaryGramSchmidt(
				pas_solver->orthX_orth_block_size,
				pas_solver->orthX_orth_max_reorth,
				pas_solver->orthX_orth_zero_tol,
				mv_ws[3][level],dbl_ws,ops_pas->app_ops);
	else 
		MultiVecOrthSetup_ModifiedGramSchmidt(
				pas_solver->orthX_orth_block_size,
				pas_solver->orthX_orth_max_reorth,
				pas_solver->orthX_orth_zero_tol,
				mv_ws[3][level],dbl_ws,ops_pas->app_ops);
		
	ops_pas->app_ops->MultiVecOrth(X,0,&endX,B[level],ops_pas->app_ops);
	//ops_pas->Printf("sizeX = %d, endX = %d\n", sizeX, endX);
	if (endX < sizeX) {
	    ops_pas->app_ops->MultiVecSetRandomValue(X,endX,sizeX,ops_pas->app_ops);
	    idx = endX; endX = sizeX;
	    ops_pas->app_ops->MultiVecOrth(X,idx,&endX,B[level],ops_pas->app_ops);
	}
	assert(sizeX == endX);

	if (level == 0) {
	   double *destin = ((LAPACKVEC*)ss_evec->x)->data+sizeX*startN;
	   start[0] = 0     ; end[0] = sizeX;
	   start[1] = startN; end[1] = sizeX;
	   ops_pas->app_ops->MultiVecQtAP('S','N',
		 X,B[level],ritz_vec,0,
		 start,end,destin,sizeX,
		 mv_ws[1][level],ops_pas->app_ops);
	}

#if DEBUG
	ops_pas->Printf("after Orth\n");
	ops_pas->app_ops->MultiVecView(X, 0, sizeX, ops_pas);
#endif
#if TIME_PAS
#if USE_MPI
    time_pas.orthX_time += MPI_Wtime();
#else
    time_pas.orthX_time += clock();
#endif
#endif
	return;
}

static void PAS(void *A, void *B , double *eval, void **evec,
		int nevGiven, int *nevConv, struct OPS_ *ops)
{
	OPS_Create (&ops_pas);
	OPS_PAS_Set(ops_pas,ops);
	OPS_Setup  (ops_pas);
    
	pas_solver = (PASSolver*)ops->eigen_solver_workspace;	

	int    nevMax, multiMax, block_size; 
	int    numIterMax, numIter, nev, numCheck;
	int    start[2], end[2];
	void   ***X, **ritz_vec;
	PASMAT ss_matA, ss_matB; PASVEC ss_evec;
	double *ss_eval, *tol; 

	ss_eval = eval; ritz_vec = evec;
	
	nevMax     = pas_solver->nevMax    ; multiMax = pas_solver->multiMax; 
	block_size = pas_solver->block_size; tol      = pas_solver->tol;
	numIterMax = pas_solver->numIterMax;

	/* 全局变量初始化 */
	sizeX = nevMax+multiMax;
	sizeC = 0; sizeN = sizeX; 
	startN = sizeC; endN = startN+sizeN;	
	level     = pas_solver->level_aux;
	level_aux = pas_solver->level_aux;

	LAPACKVEC ss_evec_x;
	ss_evec_x.nrows = ss_evec_x.ncols = ss_evec_x.ldd = sizeX;
	ss_evec_x.data  = pas_solver->dbl_ws;
	memset(ss_evec_x.data,0,sizeX*sizeX*sizeof(double));
	ss_evec.x = (void*)&ss_evec_x; 
	ss_evec.q = pas_solver->mv_ws[0];
	ss_evec.P = pas_solver->P;
	ss_evec.level_aux  = pas_solver->level_aux;
	ss_evec.num_levels = pas_solver->num_levels;
	
	LAPACKMAT ss_matA_XX;
	ss_matA_XX.nrows = ss_matA_XX.ncols = ss_matA_XX.ldd = sizeX;
	ss_matA_XX.data  = ss_evec_x.data+sizeX*sizeX;
	ss_matA.XX = (void*)&ss_matA_XX; 
	ss_matA.QX = pas_solver->mv_ws[1];
	ss_matA.QQ = pas_solver->A; 
	ss_matA.P  = pas_solver->P;
	ss_matA.level_aux  = pas_solver->level_aux;
	ss_matA.num_levels = pas_solver->num_levels;
	
	ss_matB.XX = NULL; ss_matB.QX = NULL; 
	ss_matB.QQ = pas_solver->B; 
	ss_matB.P  = pas_solver->P;
	ss_matB.level_aux  = pas_solver->level_aux;
	ss_matB.num_levels = pas_solver->num_levels;
	
	/* workspace */
	X        = ss_evec.q;
	mv_ws[0] = pas_solver->mv_ws[2];  
	mv_ws[1] = pas_solver->mv_ws[3]; 
	mv_ws[2] = pas_solver->mv_ws[4];  
	mv_ws[3] = pas_solver->mv_ws[5];
	mv_ws[4] = pas_solver->mv_ws[6]; /* sizeV */
	dbl_ws   = ss_matA_XX.data+sizeX*sizeX;
	int_ws   = pas_solver->int_ws;


#if TIME_PAS
	time_pas.checkconv_time = 0.0;
	time_pas.compN_time     = 0.0; 
	time_pas.compRR_time    = 0.0; 
	time_pas.compRV_time    = 0.0;
	time_pas.orthX_time     = 0.0;
	time_pas.promX_time     = 0.0;
#endif

	ops_pas->Printf("ComputeRayleighRitz\n");	
	ComputeRayleighRitz(&ss_matA,&ss_matB,ss_eval,&ss_evec);	

	nev = *nevConv; *nevConv = 0; numIter = 0;	
	do {

		ops_pas->Printf("------------------------------\n");
		ops_pas->Printf("level = %d, numIter = %d, sizeC = %d, sizeN = %d, sizeX = %d\n",
				level, numIter,sizeC,sizeN,sizeX);		
				
		if (level == 0) {
		
			ops_pas->Printf("CheckConvergence\n");
			numCheck = (startN+multiMax+sizeN<sizeX)?(multiMax+sizeN):(sizeX-startN);
			numCheck = numCheck<20?numCheck:20;
			sizeC = CheckConvergence(pas_solver->A[0],pas_solver->B[0],
				ss_eval,ritz_vec,numCheck,tol);

			ops_pas->Printf("%d\n",sizeC);
			if (sizeC >= nev) {
				break;
			}
			/* 当不在最细层时, 特征向量会被全算 */
			startN = sizeC; 
			endN   = startN+block_size;
			endN   = endN<sizeX?endN:sizeX;
			sizeN  = endN-startN;
		}
		else {
			ops_pas->Printf("PromoteX\n");
			/* X = ss_evec.q */
			PromoteX(X,pas_solver->P);
			--level;
			if (level == 0) {
			   start[0] = 0; end[0] = sizeX;
			   start[1] = 0; end[1] = sizeX;
			   ops_pas->app_ops->MultiVecAxpby(1.0,X[0],0.0,ritz_vec,start,end,ops_pas->app_ops);
			}
		}
				
		ops_pas->Printf("ComputeN\n");
		if (level == 0) {
		   ComputeN(ritz_vec,pas_solver->A[level],pas_solver->B[level],ss_eval);
		}
		else {
		   ComputeN(X[level],pas_solver->A[level],pas_solver->B[level],ss_eval);
		}
		
		ops_pas->Printf("OrthXtoQ\n");
		/* X = ss_evec.q */
		OrthXtoQ(&ss_evec,pas_solver->P,pas_solver->B,ritz_vec);

		ops_pas->Printf("ComputeRayleighRitz\n");
		ComputeRayleighRitz(&ss_matA,&ss_matB,ss_eval,&ss_evec);					

		ops_pas->Printf("ComputeRitzVec\n");
		if (level == 0) {
			ComputeRitzVec(ritz_vec,&ss_evec);
			
#if DEBUG
			start[0] = 0; end[0] = sizeX;
			start[1] = 0; end[1] = sizeX;
			ops_pas->Printf("VtBV\n");
			ops_pas->app_ops->MultiVecQtAP('N','N',ritz_vec,B,ritz_vec,0,start,end,
					dbl_ws,sizeX,mv_ws[0][level],ops);
			int row, col;
			for (row = 0; row < end[0]-start[0]; ++row) {
				for (col = 0; col < end[1]-start[1]; ++col) {
					ops_pas->Printf("%6.4e\t",dbl_ws[row+col*(end[0]-start[0])]);
				}	
				ops_pas->Printf("\n");
			}
#endif
			
		}
		else {
			ComputeRitzVec(X[level],&ss_evec);
		}

		++numIter;
	} while (numIter < numIterMax+pas_solver->num_levels);
	
	pas_solver->numIter = numIter;
	*nevConv = sizeC;
	
	
#if TIME_PAS
	ops_pas->Printf("|--PAS----------------------------\n");
	time_pas.time_total = time_pas.checkconv_time
		+time_pas.compN_time
		+time_pas.compRR_time
		+time_pas.compRV_time
		+time_pas.orthX_time
		+time_pas.promX_time;
	ops_pas->Printf("|checkconv  compN  compRR  compRV  orthX  promX\n");
#if USE_MPI	
	ops_pas->Printf("|%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n",
		time_pas.checkconv_time,		
		time_pas.compN_time,		
		time_pas.compRR_time,		
		time_pas.compRV_time,
		time_pas.orthX_time,
		time_pas.promX_time);	   	
#else	
	ops_pas->Printf("|%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n",
		time_pas.checkconv_time/CLOCKS_PER_SEC,		
		time_pas.compN_time    /CLOCKS_PER_SEC,		
		time_pas.compRR_time   /CLOCKS_PER_SEC,		
		time_pas.compRV_time   /CLOCKS_PER_SEC,
		time_pas.orthX_time    /CLOCKS_PER_SEC,
		time_pas.promX_time    /CLOCKS_PER_SEC);	
#endif
	ops_pas->Printf("|%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\n",
		time_pas.checkconv_time/time_pas.time_total*100,
		time_pas.compN_time    /time_pas.time_total*100,
		time_pas.compRR_time   /time_pas.time_total*100,
		time_pas.compRV_time   /time_pas.time_total*100,		
		time_pas.orthX_time    /time_pas.time_total*100,		
		time_pas.promX_time    /time_pas.time_total*100);
	ops_pas->Printf("|--PAS----------------------------\n");
	time_pas.checkconv_time = 0.0;
	time_pas.compN_time     = 0.0; 
	time_pas.compRR_time    = 0.0; 
	time_pas.compRV_time    = 0.0;
	time_pas.orthX_time     = 0.0;
	time_pas.promX_time     = 0.0;
#endif
	
	OPS_Destroy (&ops_pas);
	return;
}

/* 设定 PAS 的工作空间 */
void EigenSolverSetup_PAS(
	int    nevMax    , int    multiMax  , double gapMin,
	int    block_size, double tol[2]    , int    numIterMax,
	int block_size_rr, double tol_rr[2] , int numIterMax_rr,
	void  **A_array  , void   **B_array , void **P_array, int num_levels,
	void  ***mv_ws[6], double *dbl_ws   , int *int_ws, 
	struct OPS_ *ops)
{
	static PASSolver pas_solver_static = { 
		.nevMax     = 1   , .multiMax = 2   , .gapMin   = 0.01,
		.block_size = 1   , .tol[0]   = 1e-6, .tol[1]   = 1e-6, .numIterMax = 10, 
		.block_size_rr = 1, .tol_rr[0]= 1e-6, .tol_rr[1]= 1e-6, .numIterMax_rr = 10, 
		.mv_ws      = NULL, .dbl_ws   = NULL, .int_ws   = NULL,
		/* 算法内部参数 */	
		.compN_user_defined_multi_linear_solver = 0,
		.compN_bamg_max_iter[0]  = 1, /* bamg 最大迭代次数 */
		/* 前后光滑最大迭代次数 */
		.compN_bamg_max_iter[1]  = 4, .compN_bamg_max_iter[2]  = 4, 
		.compN_bamg_max_iter[3]  = 4, .compN_bamg_max_iter[4]  = 4, 
		.compN_bamg_max_iter[5]  = 4, .compN_bamg_max_iter[6]  = 4, 
		.compN_bamg_max_iter[7]  = 4, .compN_bamg_max_iter[8]  = 4, 
		.compN_bamg_max_iter[9]  = 4, .compN_bamg_max_iter[10] = 4, 
		.compN_bamg_max_iter[11] = 4, .compN_bamg_max_iter[12] = 4, 
		.compN_bamg_max_iter[13] = 20, .compN_bamg_max_iter[14] = 20, 
		.compN_bamg_max_iter[15] = 20, .compN_bamg_max_iter[16] = 20,
		.compN_bamg_max_iter[17] = 40, .compN_bamg_max_iter[18] = 40, 
		.compN_bamg_max_iter[19] = 40, .compN_bamg_max_iter[20] = 40, 
		.compN_bamg_max_iter[21] = 40, .compN_bamg_max_iter[22] = 40, 
		.compN_bamg_max_iter[23] = 40, .compN_bamg_max_iter[24] = 40, 
		.compN_bamg_max_iter[25] = 80, .compN_bamg_max_iter[26] = 80,
		.compN_bamg_max_iter[27] = 80, .compN_bamg_max_iter[28] = 80, 
		.compN_bamg_max_iter[29] = 80, .compN_bamg_max_iter[30] = 80, 
		/* 最粗层最大迭代次数 */
		.compN_bamg_max_iter[31] = 100,
		.compN_bamg_rate[0]  = 1e-2, .compN_bamg_rate[1]  = 1e-2,
		.compN_bamg_rate[2]  = 1e-2, .compN_bamg_rate[3]  = 1e-2,
		.compN_bamg_rate[4]  = 1e-2, .compN_bamg_rate[5]  = 1e-2,
		.compN_bamg_rate[6]  = 1e-2, .compN_bamg_rate[7]  = 1e-2,
		.compN_bamg_rate[8]  = 1e-2, .compN_bamg_rate[9]  = 1e-2,
		.compN_bamg_rate[10] = 1e-2, .compN_bamg_rate[11] = 1e-2,
		.compN_bamg_rate[12] = 1e-16, .compN_bamg_rate[13] = 1e-16,
		.compN_bamg_rate[14] = 1e-16, .compN_bamg_rate[15] = 1e-16,
		.compN_bamg_tol[0]   = 1e-14, .compN_bamg_tol[1]   = 1e-14, 
		.compN_bamg_tol[2]   = 1e-14, .compN_bamg_tol[3]   = 1e-14, 
		.compN_bamg_tol[4]   = 1e-14, .compN_bamg_tol[5]   = 1e-14, 
		.compN_bamg_tol[6]   = 1e-14, .compN_bamg_tol[7]   = 1e-14, 
		.compN_bamg_tol[8]   = 1e-26, .compN_bamg_tol[9]   = 1e-26, 
		.compN_bamg_tol[10]  = 1e-26, .compN_bamg_tol[11]  = 1e-26, 
		.compN_bamg_tol[12]  = 1e-26, .compN_bamg_tol[13]  = 1e-26, 
		.compN_bamg_tol[14]  = 1e-26, .compN_bamg_tol[15]  = 1e-26, 
		.compN_bamg_tol_type = "abs",
		.orthX_user_defined_multi_linear_solver = 0,
		.orthX_ls_max_iter                = 4    ,
		.orthX_ls_rate                    = 1e-2 ,
		.orthX_ls_tol                     = 1e-14, 
		.orthX_ls_tol_type                = "abs",
		.orthX_orth_method                = "bgs", 
		.orthX_orth_block_size            = -1   ,
		.orthX_orth_max_reorth            = 4    ,
		.orthX_orth_zero_tol              = 1e-16,
		.compRR_gcg_check_conv_max_num    = 20   ,
		.compRR_gcg_initX_orth_method     = "bgs",
		.compRR_gcg_initX_orth_block_size = -1   ,
		.compRR_gcg_initX_orth_max_reorth = 4    ,
		.compRR_gcg_initX_orth_zero_tol   = 1e-16,
		.compRR_gcg_compP_orth_method     = "bgs",
		.compRR_gcg_compP_orth_block_size = -1   ,
		.compRR_gcg_compP_orth_max_reorth = 4    ,
		.compRR_gcg_compP_orth_zero_tol   = 1e-16,
		.compRR_gcg_compW_orth_method     = "bgs",
		.compRR_gcg_compW_orth_block_size = -1   ,
		.compRR_gcg_compW_orth_max_reorth = 4    ,
		.compRR_gcg_compW_orth_zero_tol   = 1e-16,
		.compRR_gcg_compW_cg_max_iter     = 10   ,
		.compRR_gcg_compW_cg_rate         = 1e-2 ,
		.compRR_gcg_compW_cg_tol          = 1e-14,
		.compRR_gcg_compW_cg_tol_type     = "abs",
		.compRR_gcg_compRR_min_num        = -1   ,
		.compRR_gcg_compRR_min_gap        = 0.01 ,
		.compRR_gcg_compRR_tol            = 1e-16,		
	};
	if (nevMax>0)	
		pas_solver_static.nevMax     = nevMax;
	if (multiMax>=0)
		pas_solver_static.multiMax   = multiMax;
	if (gapMin>=0)
		pas_solver_static.gapMin     = gapMin;
	if (block_size>0)
		pas_solver_static.block_size = block_size;
	if (tol!=NULL) {
		pas_solver_static.tol[0]     = tol[0];
		pas_solver_static.tol[1]     = tol[1];
	}
	if (numIterMax>0)
		pas_solver_static.numIterMax = numIterMax;
	if (block_size_rr>0)
		pas_solver_static.block_size_rr = block_size_rr;
	if (tol_rr!=NULL) {
		pas_solver_static.tol_rr[0]  = tol_rr[0];
		pas_solver_static.tol_rr[1]  = tol_rr[1];	
	}
	if (numIterMax_rr>0)
		pas_solver_static.numIterMax_rr = numIterMax_rr;
	
	pas_solver_static.A          = A_array;
	pas_solver_static.B          = B_array;
	pas_solver_static.P          = P_array;
	pas_solver_static.num_levels = num_levels;
	/* 可更改 >0 即可 */
	pas_solver_static.level_aux  = num_levels-1;
	
	pas_solver_static.mv_ws[0]   = mv_ws[0];
	pas_solver_static.mv_ws[1]   = mv_ws[1];
	pas_solver_static.mv_ws[2]   = mv_ws[2];
	pas_solver_static.mv_ws[3]   = mv_ws[3];
	pas_solver_static.mv_ws[4]   = mv_ws[4];
	pas_solver_static.mv_ws[5]   = mv_ws[5];
	pas_solver_static.mv_ws[6]   = mv_ws[6];
	pas_solver_static.dbl_ws     = dbl_ws;
 	pas_solver_static.int_ws     = int_ws;
		
	ops->eigen_solver_workspace = (void *)(&pas_solver_static);
	ops->EigenSolver            = PAS;
	
	return;	
}


/* 参数设定函数需要在 Setup 之后调用 */
void EigenSolverSetParameters_PAS(
	int     check_conv_max_num ,
	int     compN_user_defined_multi_linear_solver,
	int    *compN_bamg_max_iter, double *compN_bamg_rate,
	double *compN_bamg_tol     , const char *compN_bamg_tol_type,
	int     orthX_user_defined_multi_linear_solver,
	int     orthX_ls_max_iter  , double  orthX_ls_rate,
	double  orthX_ls_tol       , const char *orthX_ls_tol_type,
	const char *orthX_orth_method, int   orthX_orth_block_size, int orthX_orth_max_reorth,
	double  orthX_orth_zero_tol,
	int     compRR_gcg_check_conv_max_num,
	const char *compRR_gcg_initX_orth_method,
	int     compRR_gcg_initX_orth_block_size,
	int     compRR_gcg_initX_orth_max_reorth,
	double  compRR_gcg_initX_orth_zero_tol,
	const char *compRR_gcg_compP_orth_method,
	int     compRR_gcg_compP_orth_block_size,
	int     compRR_gcg_compP_orth_max_reorth,
	double  compRR_gcg_compP_orth_zero_tol,
	const char *compRR_gcg_compW_orth_method,
	int     compRR_gcg_compW_orth_block_size,
	int     compRR_gcg_compW_orth_max_reorth,
	double  compRR_gcg_compW_orth_zero_tol,
	int     compRR_gcg_compW_cg_max_iter,
	double  compRR_gcg_compW_cg_rate,
	double  compRR_gcg_compW_cg_tol,
	const char *compRR_gcg_compW_cg_tol_type,
	int     compRR_gcg_compRR_min_num,
	double  compRR_gcg_compRR_min_gap,
	double  compRR_gcg_compRR_tol,
	struct  OPS_ *ops)
{
	/* 当前版本不支持外部求解器 */
	compN_user_defined_multi_linear_solver = 0;
	orthX_user_defined_multi_linear_solver = 0;
	int level;
	struct PASSolver_ *pas_solver = (PASSolver*)ops->eigen_solver_workspace;
	pas_solver->compN_user_defined_multi_linear_solver = compN_user_defined_multi_linear_solver;
	if (compN_bamg_max_iter!=NULL) {
		/* bamg 最大迭代次数 */
		pas_solver->compN_bamg_max_iter[0] = compN_bamg_max_iter[0];
		for (level = 0; level < pas_solver->num_levels-1; ++level) {
			/* 前光滑 最大迭代次数 */
			pas_solver->compN_bamg_max_iter[2*level+1] = compN_bamg_max_iter[2*level+1];
			/* 后光滑 最大迭代次数 */
			pas_solver->compN_bamg_max_iter[2*level+2] = compN_bamg_max_iter[2*level+2];
		}
		/* 最粗层 最大迭代次数 */
		pas_solver->compN_bamg_max_iter[2*(pas_solver->num_levels-1)+1] 
			= compN_bamg_max_iter[2*(pas_solver->num_levels-1)+1];
	}
	if (compN_bamg_rate!=NULL) {
		for (level = 0; level < pas_solver->num_levels; ++level) {
			/* 每层的收敛率 */
			pas_solver->compN_bamg_rate[level] = compN_bamg_rate[level];
		}
	}
	if (compN_bamg_tol!=NULL) {
		for (level = 0; level < pas_solver->num_levels; ++level) {
			/* 每层的误差限 */
			pas_solver->compN_bamg_tol[level] = compN_bamg_tol[level];
		}
	}
	pas_solver->compN_bamg_tol_type = compN_bamg_tol_type;

	pas_solver->orthX_user_defined_multi_linear_solver = orthX_user_defined_multi_linear_solver;
	/* 线性求解器 默认选用 AMG */
	if (orthX_ls_max_iter>0)
		pas_solver->orthX_ls_max_iter = orthX_ls_max_iter;
	if (orthX_ls_rate>0)
		pas_solver->orthX_ls_rate     = orthX_ls_rate;
	if (orthX_ls_tol>0)
		pas_solver->orthX_ls_tol      = orthX_ls_tol;
	pas_solver->orthX_ls_tol_type = orthX_ls_tol_type;

	pas_solver->orthX_orth_method = orthX_orth_method;
	if (orthX_orth_block_size>0)
		pas_solver->orthX_orth_block_size = orthX_orth_block_size;
	if (orthX_orth_max_reorth>=0)
		pas_solver->orthX_orth_max_reorth = orthX_orth_max_reorth;
	if (orthX_orth_zero_tol>0)
		pas_solver->orthX_orth_zero_tol	  = orthX_orth_zero_tol;
	
	pas_solver->compRR_gcg_check_conv_max_num    = compRR_gcg_check_conv_max_num;
	pas_solver->compRR_gcg_initX_orth_method     = compRR_gcg_initX_orth_method;
	pas_solver->compRR_gcg_initX_orth_block_size = compRR_gcg_initX_orth_block_size;
	pas_solver->compRR_gcg_initX_orth_zero_tol   = compRR_gcg_initX_orth_zero_tol;
	pas_solver->compRR_gcg_compP_orth_method     = compRR_gcg_compP_orth_method;
	pas_solver->compRR_gcg_compP_orth_block_size = compRR_gcg_compP_orth_block_size;
	pas_solver->compRR_gcg_compP_orth_zero_tol   = compRR_gcg_compP_orth_zero_tol;
	pas_solver->compRR_gcg_compW_orth_method     = compRR_gcg_compW_orth_method;
	pas_solver->compRR_gcg_compW_orth_block_size = compRR_gcg_compW_orth_block_size;
	pas_solver->compRR_gcg_compW_orth_zero_tol   = compRR_gcg_compW_orth_zero_tol;
	pas_solver->compRR_gcg_compW_cg_max_iter     = compRR_gcg_compW_cg_max_iter;
	pas_solver->compRR_gcg_compW_cg_rate         = compRR_gcg_compW_cg_rate;
	pas_solver->compRR_gcg_compW_cg_tol          = compRR_gcg_compW_cg_tol;
	pas_solver->compRR_gcg_compW_cg_tol_type     = compRR_gcg_compW_cg_tol_type;
	pas_solver->compRR_gcg_compRR_min_num        = compRR_gcg_compRR_min_num;
	pas_solver->compRR_gcg_compRR_min_gap        = compRR_gcg_compRR_min_gap;
	pas_solver->compRR_gcg_compRR_tol            = compRR_gcg_compRR_tol;	

	return;
}
