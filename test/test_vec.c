/**
 *    @file  vec.c
 *   @brief  test vec in ops
 *
 *  测试lapack app中的矩阵向量操作
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/13
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>

#include    "ops.h"

int TestVec (void *mat, struct OPS_ *ops) 
{
	ops->Printf("TestVec\n");
	srand(0);
	double inner_prod = 0, alpha = 2.0, beta = 2.0;
	void *vec[2];
	ops->Printf("mat:\n");
	ops->MatView(mat,ops);
	ops->VecCreateByMat(&(vec[0]),mat,ops);
	ops->VecCreateByMat(&(vec[1]),mat,ops);
	ops->VecSetRandomValue(vec[0],ops);
	ops->VecSetRandomValue(vec[1],ops);
	ops->Printf("vec x:\n");
	ops->VecView(vec[0],ops);
	ops->Printf("vec y:\n");
	ops->VecView(vec[1],ops);
	ops->VecInnerProd(vec[0],vec[1],&inner_prod,ops);
	ops->Printf("x^t y = %6.4e\n",inner_prod);
	ops->Printf("y = %f x + %f y\n",alpha,beta);
	ops->VecAxpby(alpha,vec[0],beta,vec[1],ops);
	ops->VecView(vec[1],ops);
	ops->Printf("y = A x\n");
	ops->MatDotVec(mat,vec[0],vec[1],ops);
	ops->VecView(vec[1],ops);
	ops->Printf("y = A^t x\n");
	ops->MatTransDotVec(mat,vec[0],vec[1],ops);
	ops->VecView(vec[1],ops);

	ops->VecDestroy(&(vec[0]),ops);
	ops->VecDestroy(&(vec[1]),ops);
	return 0;
}

