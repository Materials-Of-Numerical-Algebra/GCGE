/**
 *    @file  ops_lin_sol.h
 *   @brief  operations for linear solver 
 *
 *  ���������
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/15
 *      Revision:  none
 */
#ifndef  _OPS_LIN_SOL_H_
#define  _OPS_LIN_SOL_H_

#include	"ops.h"
#include	"app_lapack.h"

typedef struct PCGSolver_ {
	int max_iter; double rate; double tol; const char *tol_type;
	void *vec_ws[3];  /* r p w */
	void *pc;
	int niter; double residual;
}PCGSolver;

void LinearSolverSetup_PCG(
	int max_iter, double rate, double tol, const char *tol_type, 
	void *vec_ws[3], void *pc, struct OPS_ *ops);
			
typedef struct BlockPCGSolver_ {
	int max_iter; double rate; double tol; const char *tol_type;
	void   **mv_ws[3]; /* r p w */
	double *dbl_ws; /* (6*length of vec) */
	int    *int_ws; /* (2*length of vec) */
	void   *pc;
	int niter; double residual;
}BlockPCGSolver;
void MultiLinearSolverSetup_BlockPCG(
	int max_iter, double rate, double tol, const char *tol_type,
	void   **mv_ws[3], double *dbl_ws, int *int_ws,
	void   *pc, struct OPS_ *ops);	
typedef struct BlockAMGSolver_ {
    int  *max_iter; double *rate; double *tol; const char *tol_type;
    void **A_array; void **P_array; int num_levels;
    void ***mv_array_ws[5]; double *dbl_ws; int *int_ws; 
    void *pc;
    int  niter; double residual;
}BlockAMGSolver;
void MultiLinearSolverSetup_BlockAMG(
		int *max_iter, double *rate, double *tol, const char *tol_type, 
		void **A_array, void **P_array, int num_levels,
		void ***mv_array_ws[5], double *dbl_ws, int *int_ws,
		void *pc, struct OPS_ *ops);

#endif  /* -- #ifndef _OPS_LIN_SOL_H_ -- */


