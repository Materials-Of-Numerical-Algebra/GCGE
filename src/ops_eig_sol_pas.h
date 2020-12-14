/**
 *    @file  ops_eig_sol_pas.h
 *   @brief  GCG ����ֵ����� 
 *
 *  GCG ����ֵ�����
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/29
 *      Revision:  none
 */
#ifndef  _OPS_EIG_SOL_PAS_H_
#define  _OPS_EIG_SOL_PAS_H_

#include    "ops.h"
#include    "ops_orth.h"
#include    "ops_lin_sol.h"
#include    "ops_eig_sol_gcg.h"
#include    "app_lapack.h"
#include    "app_pas.h"

/* PAS �Ľṹ�� */
typedef struct PASSolver_ {
	void   **A       ; void  **B     ; void **P;
	double *eval     ; void  **evec  ; 
	int    nevMax    ; int   multiMax; double gapMin;
	int    block_size; double tol[2] ; int numIterMax;
	int    num_levels; int  level_aux; 	
	int    numIter   ;
	/* for gcg in ComputeRayleighRitz */
	int block_size_rr; double tol_rr[2]; int numIterMax_rr;
	/* workspace */
	void  ***mv_ws[7]; double *dbl_ws; int *int_ws;
	/* �㷨�ڲ����� */
	/* ��֧��ʹ���ⲿ����� */
	int    check_conv_max_num;
	int    compN_user_defined_multi_linear_solver;
	int    compN_bamg_max_iter[32]; 
	double compN_bamg_rate[16];
	double compN_bamg_tol[16]; 
	const char *compN_bamg_tol_type;
	/* ��֧��ʹ���ⲿ����� */
	int    orthX_user_defined_multi_linear_solver;
	int    orthX_ls_max_iter; 
	double orthX_ls_rate;
	double orthX_ls_tol; 
	const char *orthX_ls_tol_type;
	const char *orthX_orth_method; 
	int    orthX_orth_block_size;
	int    orthX_orth_max_reorth;
	double orthX_orth_zero_tol;
	int    compRR_gcg_check_conv_max_num;
	const char *compRR_gcg_initX_orth_method;
	int    compRR_gcg_initX_orth_block_size;
	int    compRR_gcg_initX_orth_max_reorth;
	double compRR_gcg_initX_orth_zero_tol;
	const char *compRR_gcg_compP_orth_method;
	int    compRR_gcg_compP_orth_block_size;
	int    compRR_gcg_compP_orth_max_reorth;
	double compRR_gcg_compP_orth_zero_tol;
	const char *compRR_gcg_compW_orth_method;
	int    compRR_gcg_compW_orth_block_size;
	int    compRR_gcg_compW_orth_max_reorth;
	double compRR_gcg_compW_orth_zero_tol;
	int    compRR_gcg_compW_cg_max_iter;
	double compRR_gcg_compW_cg_rate;
	double compRR_gcg_compW_cg_tol;
	const char *compRR_gcg_compW_cg_tol_type;
	int    compRR_gcg_compRR_min_num;
	double compRR_gcg_compRR_min_gap;
	double compRR_gcg_compRR_tol;
	//struct OPS_ *pas_ops; 
} PASSolver;

/* �趨 PAS �Ĺ����ռ� */
void EigenSolverSetup_PAS(
	int    nevMax    , int    multiMax , double gapMin,
	int    block_size, double tol[2]   , int    numIterMax,
	int block_size_rr, double tol_rr[2], int  numIterMax_rr,
	void  **A_array  , void   **B_array, void **P_array, 
	int   num_levels ,
	void  ***mv_ws[7], double *dbl_ws  , int  *int_ws, 
	struct OPS_ *ops);


/* �����趨������Ҫ�� Setup ֮����� */
void EigenSolverSetParameters_PAS(
	int     check_conv_max_num, 
	int     compN_user_define_multi_linear_solver,
	int    *compN_bamg_max_iter, double *compN_bamg_rate,
	double *compN_bamg_tol     , const char *compN_bamg_tol_type,
	int     orthX_user_define_multi_linear_solver,
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
	struct  OPS_ *ops); 


#endif  /* -- #ifndef _OPS_EIG_SOL_GCG_H_ -- */

