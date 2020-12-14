/**
 *    @file  multi_grid.c
 *   @brief  test multi-grid in ops
 *
 *  测试lapack app中的矩阵多向量操作
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/13
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>

#include    "ops.h"
#include    "ops_lin_sol.h"

int TestMultiGrid(void *A, void *B, struct OPS_ *ops) 
{
	ops->Printf("TestMultiGrid\n");
	int num_vec = 3, start[2], end[2], level;
	void **multi_vec_x[3] = {NULL,NULL,NULL};
	void **multi_vec_y[3] = {NULL,NULL,NULL};
	void **multi_vec_z[3] = {NULL,NULL,NULL};
	void *matA = A, *matB = B;
	srand(0);
		
	ops->Printf("MultiGridCreate\n");
	void **A_array, **B_array, **P_array; int num_levels = 2;
	ops->MultiGridCreate (&A_array,&B_array,&P_array,&num_levels,matA,matB,ops);
	
	for (level = 0; level < num_levels; ++level) {
		ops->MultiVecCreateByMat(&(multi_vec_x[level]),num_vec,A_array[level],ops);
		ops->MultiVecCreateByMat(&(multi_vec_y[level]),num_vec,A_array[level],ops);
		ops->MultiVecCreateByMat(&(multi_vec_z[level]),num_vec,A_array[level],ops);
		ops->MultiVecSetRandomValue(multi_vec_x[level],0,num_vec,ops);
		ops->MultiVecSetRandomValue(multi_vec_y[level],0,num_vec,ops);
		ops->MultiVecSetRandomValue(multi_vec_z[level],0,num_vec,ops);
	}

	for (level = 0; level < num_levels; ++level) {
		ops->Printf("A[%d]:\n",level);
		ops->MatView(A_array[level],ops);
		ops->Printf("B[%d]:\n",level);
		ops->MatView(B_array[level],ops);
	}
	for (level = 0; level < num_levels-1; ++level) {
		ops->Printf("P[%d]:\n",level);
		ops->MatView(P_array[level],ops);
	}
	ops->Printf("multi vec x:\n");
	ops->MultiVecView(multi_vec_x[num_levels-1],0,num_vec,ops);
	ops->Printf("multi vec y:\n");
	ops->MultiVecView(multi_vec_y[num_levels-1],0,num_vec,ops);
	ops->Printf("y = A x\n");
	start[0] = 0; end[0] = num_vec;
	start[1] = 0; end[1] = num_vec;
	ops->MatDotMultiVec(A_array[num_levels-1],
	      multi_vec_x[num_levels-1],multi_vec_y[num_levels-1],start,end,ops);
	ops->Printf("from %d to 0\n", num_levels-1);
	ops->MultiVecView(multi_vec_y[num_levels-1],start[1],end[1],ops);	
	ops->MultiVecFromItoJ(P_array,num_levels-1,0,multi_vec_y[num_levels-1],multi_vec_y[0],
			start,end,multi_vec_z,ops);
	ops->Printf("multi vec y[%d]:\n", num_levels-1);
	ops->MultiVecView(multi_vec_y[0],0,num_vec,ops);
	ops->Printf("from 0 to %d\n", num_levels-1); 
	ops->Printf("multi vec y[0]:\n");
	ops->MultiVecView(multi_vec_y[0],0,num_vec,ops);
	ops->MultiVecFromItoJ(P_array,0,num_levels-1,multi_vec_y[0],multi_vec_y[num_levels-1],
			start,end,multi_vec_z,ops);
	ops->Printf("multi vec y[%d]:\n", num_levels-1);
	ops->MultiVecView(multi_vec_y[num_levels-1],0,num_vec,ops);
	//TestEigenSolver(A_array[0],B_array[0],ops);
	//TestEigenSolver(A_array[1],B_array[1],ops);
	//TestEigenSolver(A_array[2],B_array[2],ops);
	for (level = 0; level < num_levels; ++level) {
		ops->MultiVecDestroy(&(multi_vec_z[level]),num_vec,ops);
	}
	/* 测试 AMG */ 
	/* max_iter[level*2] 前光滑次数, max_iter[level*2+1] 光滑次数
	 * max_iter[num_levels*2] 表示 V cycle 的最大次数 */ 
	//int    max_iter[6] = {8,10,30,100,100,100}, idx;
	int    max_iter[6] = {1,5,1,0,100,100}, idx;
	/* tol[num_levels] 表示 V cycle 的停止准则, 绝对残量 */ 
	double rate[3] = {1e-36,1e-36,1e-36}, tol[4] = {1e-36,1e-36,1e-36,1e-36}; 
	void   ***mv_ws[5] = {NULL}; 
	double dbl_ws[4096] = {0.0}; int int_ws[1024] = {0};	
	for (idx = 0; idx < 5; ++idx) {
		mv_ws[idx] = malloc(num_levels*sizeof(void**));
		for (level = 0; level < num_levels; ++level) {
			ops->MultiVecCreateByMat(&(multi_vec_z[level]),num_vec,A_array[level],ops);
			ops->MultiVecSetRandomValue(multi_vec_z[level],0,num_vec,ops);
			mv_ws[idx][level] = multi_vec_z[level];
		}
	}
	//mv_ws[4] = mv_ws[0];
	
	start[0] = 0; end[0] = num_vec;
	start[1] = 0; end[1] = num_vec;
	//ops->MultiVecSetRandomValue(multi_vec_x[0],start[0],end[0],ops);
	ops->MultiVecAxpby(0.0,NULL,0.0,multi_vec_x[0],start,end,ops);
	ops->MatDotMultiVec(A_array[0],multi_vec_x[0],multi_vec_y[0],start,end,ops);
	ops->Printf("Exact Solution:\n");
	ops->MultiVecView(multi_vec_x[0],0,num_vec,ops);
	//ops->MultiVecSetConstValue(multi_vec[0],0,num_vec,1.0,ops);
	double *data = ((LAPACKVEC*)multi_vec_x[0])->data;
	int    nrows = ((LAPACKVEC*)multi_vec_x[0])->nrows;
	int    ldd   = ((LAPACKVEC*)multi_vec_x[0])->ldd;
	for (idx = 0; idx < ldd*num_vec;++idx) {
		data[idx] = 1;
	}	
	//ops->MultiVecSetRandomValue(multi_vec_x[0],start[0],end[0],ops);
	ops->MultiVecAxpby(0.0,NULL,1e0,multi_vec_x[0],start,end,ops);	
	MultiLinearSolverSetup_BlockAMG(max_iter, rate, tol,
		"abs", A_array, P_array, num_levels, 
		mv_ws, dbl_ws, int_ws, NULL, ops);
	ops->Printf("V-cycle\n");	
	ops->MultiLinearSolver(A_array[0],multi_vec_y[0],multi_vec_x[0],start,end,ops);
	//ops->Printf("V-cycle\n");
	//ops->MultiLinearSolver(A_array[0],multi_vec_y[0],multi_vec_x[0],start,end,ops);
	ops->Printf("Numerical Solution:\n");
	ops->MultiVecView(multi_vec_x[0],0,num_vec,ops);	
	
	for (idx = 0; idx < 5; ++idx) {		
		for (level = 0; level < num_levels; ++level) {
			ops->MultiVecDestroy(&(mv_ws[idx][level]),num_vec,ops);
		}
		free(mv_ws[idx]);
	}
	for (level = 0; level < num_levels; ++level) {
		ops->MultiVecDestroy(&(multi_vec_x[level]),num_vec,ops);
		ops->MultiVecDestroy(&(multi_vec_y[level]),num_vec,ops);
	}
	ops->Printf("MultiGridDestroy\n");
	ops->MultiGridDestroy(&A_array,&B_array,&P_array,&num_levels,ops);


	return 0;
}
