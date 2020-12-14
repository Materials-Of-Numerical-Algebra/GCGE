/**
 *    @file  lin_sol.c
 *   @brief  linear solver 
 *
 *  测试线性求解器
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/15
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include	<memory.h>

#include    "ops.h"
#include    "ops_lin_sol.h"

int TestLinearSolver(void *mat, struct OPS_ *ops) 
{
	srand(0);
	void *vec[5]; /* b x r p w */ 
	double rate, tol; int max_iter, idx;
	for (idx = 0; idx < 4; ++idx) {
		ops->VecCreateByMat(&(vec[idx]),mat,ops);
	}
	vec[4] = vec[0]; /* using b as w */
	ops->VecSetRandomValue(vec[1],ops);
	ops->MatDotVec(mat,vec[1],vec[0],ops);
	ops->Printf("mat A:\n");
	ops->MatView(mat,ops);
	ops->Printf("vec b:\n");
	ops->VecView(vec[0],ops);
	ops->Printf("vec x:\n");
	ops->VecView(vec[1],ops);
	ops->VecSetRandomValue(vec[1],ops);
	/* PCG */
	max_iter = 50; rate = 1e-16; tol = 1e-12;
	LinearSolverSetup_PCG(max_iter,rate,tol,"rel",vec+2,NULL,ops);
	ops->LinearSolver(mat,vec[0],vec[1],ops);
	ops->Printf("niter = %d, residual = %6.4e\n",
		((PCGSolver*)(ops->linear_solver_workspace))->niter,
		((PCGSolver*)(ops->linear_solver_workspace))->residual);
	ops->Printf("vec x:\n");
	ops->VecView(vec[1],ops);
	ops->Printf("b - A x\n");
	ops->MatDotVec(mat,vec[1],vec[2],ops);
	ops->VecAxpby(-1.0,vec[2],1.0,vec[0],ops);
	ops->VecView(vec[0],ops);

	for (idx = 0; idx < 4; ++idx) {
		ops->VecDestroy(&(vec[idx]),ops);
	}
	return 0;
}

int TestMultiLinearSolver(void *mat, struct OPS_ *ops) 
{
	srand(0);
	void **multi_vec[5]; /* b x r p w */ 
	double rate, tol, dbl_ws[1024]; 
	int max_iter, idx, num_vec = 4, start[2], end[2], int_ws[256];
	
	memset(dbl_ws,0,1024*sizeof(double));
	memset(int_ws,0,256*sizeof(int));
	
	for (idx = 0; idx < 4; ++idx) {
		ops->MultiVecCreateByMat(&(multi_vec[idx]),num_vec,mat,ops);
	}
	multi_vec[4] = multi_vec[0];
	ops->MultiVecSetRandomValue(multi_vec[0],0,num_vec,ops);
	ops->MultiVecSetRandomValue(multi_vec[1],0,num_vec,ops);
	ops->Printf("mat A:\n");
	ops->MatView(mat,ops);
	ops->Printf("multi vec x:\n");
	ops->MultiVecView(multi_vec[1],0,num_vec,ops);
	start[0] = 0; end[0] = num_vec;
	start[1] = 0; end[1] = num_vec;
	ops->MatDotMultiVec(mat,multi_vec[1],multi_vec[0],start,end,ops);
	ops->MultiVecSetRandomValue(multi_vec[1],0,num_vec,ops);
	/* PCG */
	max_iter = 50; rate = 1e-8; tol = 1e-8;
	MultiLinearSolverSetup_BlockPCG(max_iter,rate,tol,"abs",
			multi_vec+2,dbl_ws,int_ws,NULL,ops);
	start[0] = 0; end[0] = num_vec;
	start[1] = 0; end[1] = num_vec;
	//ops->MultiVecAxpby(0.0,NULL,0.0,multi_vec[1],start,end,ops);
	ops->MultiLinearSolver(mat,multi_vec[0],multi_vec[1],start,end,ops);
	ops->Printf("niter = %d, residual = %6.4e\n",
		((BlockPCGSolver*)(ops->multi_linear_solver_workspace))->niter,
		((BlockPCGSolver*)(ops->multi_linear_solver_workspace))->residual);
	ops->Printf("multi vec x:\n");
	ops->MultiVecView(multi_vec[1],0,num_vec,ops);
	ops->MatDotMultiVec(mat,multi_vec[1],multi_vec[0],start,end,ops);
	ops->MultiVecSetRandomValue(multi_vec[1],num_vec-2,num_vec,ops);
	ops->Printf("multi vec x:\n");
	ops->MultiVecView(multi_vec[1],0,num_vec,ops);
	ops->MultiLinearSolver(mat,multi_vec[0],multi_vec[1],start,end,ops);
	ops->Printf("niter = %d, residual = %6.4e\n",
		((BlockPCGSolver*)(ops->multi_linear_solver_workspace))->niter,
		((BlockPCGSolver*)(ops->multi_linear_solver_workspace))->residual);
	ops->Printf("multi vec x:\n");
	ops->MultiVecView(multi_vec[1],0,num_vec,ops);
	//ops->Printf("b - A x\n");
	//ops->MatDotMultiVec(mat,multi_vec[1],multi_vec[2],start,end,ops);
	//ops->MultiVecAxpby(-1.0,multi_vec[2],1.0,multi_vec[0],start,end,ops);
	//ops->MultiVecView(multi_vec[0],0,num_vec,ops);

	for (idx = 0; idx < 4; ++idx) {
		ops->MultiVecDestroy(&(multi_vec[idx]),num_vec,ops);
	}
	return 0;
}


