/**
 *    @file  multi_vec.c
 *   @brief  test multi-vec in ops
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
#include	<memory.h>

#include    "ops.h"

int TestMultiVec(void *mat, struct OPS_ *ops) 
{
	ops->Printf("TestMultiVec\n");
	double inner_prod[1024], alpha = 2.0, beta = 2.0;
	int num_vec_x = 2, num_vec_y = 5, start[2], end[2];
	int row, col;
	void **multi_vec[3] = {NULL,NULL,NULL};
	
	memset(inner_prod,0,1024*sizeof(double));
	
	srand(0);
	ops->MultiVecCreateByMat(&(multi_vec[0]),num_vec_x,mat,ops);
	ops->MultiVecCreateByMat(&(multi_vec[1]),num_vec_y,mat,ops);
	ops->MultiVecCreateByMat(&(multi_vec[2]),num_vec_x+num_vec_y,mat,ops);
	ops->MultiVecSetRandomValue(multi_vec[0],0,num_vec_x,ops);
	ops->MultiVecSetRandomValue(multi_vec[1],0,num_vec_y,ops);
	ops->Printf("multi vec x:\n");
	ops->MultiVecView(multi_vec[0],0,num_vec_x,ops);
	ops->Printf("multi vec y:\n");
	ops->MultiVecView(multi_vec[1],0,num_vec_y,ops);
	ops->Printf("D: x^t x = \n");
	start[0] = 1; end[0] = num_vec_x;
	start[1] = 1; end[1] = num_vec_x;
	ops->MultiVecInnerProd('D',multi_vec[0],multi_vec[0],0,
		start,end,inner_prod,1,ops);
	for (row = 0; row < end[0]-start[0]; ++row) {
		ops->Printf("%6.4e\t",inner_prod[row]);
	}
	ops->Printf("\n");
	ops->Printf("S: x^t x = \n");
	start[0] = 0; end[0] = num_vec_x;
	start[1] = 0; end[1] = num_vec_x;
	ops->MultiVecInnerProd('S',multi_vec[0],multi_vec[0],0,
		start,end,inner_prod,end[0]-start[0],ops);
	for (row = 0; row < end[0]-start[0]; ++row) {
		for (col = 0; col < end[1]-start[1]; ++col) {
			ops->Printf("%6.4e\t",inner_prod[row+col*(end[0]-start[0])]);
		}
		ops->Printf("\n");
	}
	ops->Printf("N: y^t y = \n");
	start[0] = 0; end[0] = num_vec_y;
	start[1] = 0; end[1] = num_vec_y;
	ops->MultiVecInnerProd('N',multi_vec[1],multi_vec[1],0,
		start,end,inner_prod,end[0]-start[0],ops);
	for (row = 0; row < end[0]-start[0]; ++row) {
		for (col = 0; col < end[1]-start[1]; ++col) {
			ops->Printf("%6.4e\t",inner_prod[row+col*(end[0]-start[0])]);
		}
		ops->Printf("\n");
	}
	ops->Printf("S: y^t y = \n");
	start[0] = 1; end[0] = num_vec_y;
	start[1] = 1; end[1] = num_vec_y;
	ops->MultiVecInnerProd('S',multi_vec[1],multi_vec[1],0,
		start,end,inner_prod,end[0]-start[0],ops);
	for (row = 0; row < end[0]-start[0]; ++row) {
		for (col = 0; col < end[1]-start[1]; ++col) {
			ops->Printf("%6.4e\t",inner_prod[row+col*(end[0]-start[0])]);
		}
		ops->Printf("\n");
	}
	ops->Printf("N: x^t y = \n");
	start[0] = 0; end[0] = num_vec_x;
	start[1] = 0; end[1] = num_vec_y;
	ops->MultiVecInnerProd('N',multi_vec[0],multi_vec[1],0,
		start,end,inner_prod,end[0]-start[0],ops);
	for (row = 0; row < end[0]-start[0]; ++row) {
		for (col = 0; col < end[1]-start[1]; ++col) {
			ops->Printf("%6.4e\t",inner_prod[row+col*(end[0]-start[0])]);
		}
		ops->Printf("\n");
	}
	ops->Printf("N: x^t y = \n");
	start[0] = 1; end[0] = num_vec_x;
	start[1] = 1; end[1] = num_vec_y;
	ops->MultiVecInnerProd('N',multi_vec[0],multi_vec[1],0,
		start,end,inner_prod,end[0]-start[0],ops);
	for (row = 0; row < end[0]-start[0]; ++row) {
		for (col = 0; col < end[1]-start[1]; ++col) {
			ops->Printf("%6.4e\t",inner_prod[row+col*(end[0]-start[0])]);
		}
		ops->Printf("\n");
	}
	ops->Printf("y = %f x + %f y\n",alpha,beta);
	start[0] = 1; end[0] = num_vec_x;
	start[1] = 1; end[1] = num_vec_x;
	ops->MultiVecAxpby(alpha,multi_vec[0],beta,multi_vec[1],
		start,end,ops);
	ops->MultiVecView(multi_vec[1],start[1],end[1],ops);

	ops->Printf("x[1:2] = x[0:1]\n");
	alpha = 1.0; beta = 0.0;
	start[0] = 0; end[0] = 1;
	start[1] = 1; end[1] = 2;
	ops->MultiVecAxpby(alpha,multi_vec[0],beta,multi_vec[0],
		start,end,ops);
	ops->MultiVecView(multi_vec[0],0,2,ops);

	ops->Printf("mat A:\n");
	ops->MatView(mat,ops);
	ops->Printf("y = A x\n");
	start[0] = 1; end[0] = num_vec_x;
	start[1] = 1; end[1] = num_vec_x;
	ops->MatDotMultiVec(mat,multi_vec[0],multi_vec[1],
		start,end,ops);
	ops->MultiVecView(multi_vec[1],start[1],end[1],ops);
	ops->Printf("y = A x\n");
	start[0] = 0; end[0] = num_vec_x;
	start[1] = 0; end[1] = num_vec_x;
	ops->MatDotMultiVec(mat,multi_vec[0],multi_vec[1],
		start,end,ops);
	ops->MultiVecView(multi_vec[1],start[1],end[1],ops);
	ops->Printf("y = A^t x\n");
	start[0] = 1; end[0] = num_vec_x;
	start[1] = 1; end[1] = num_vec_x;
	ops->MatTransDotMultiVec(mat,multi_vec[0],multi_vec[1],
		start,end,ops);
	ops->MultiVecView(multi_vec[1],start[1],end[1],ops);
	
	double *qAp; int ldQAP;
	ops->Printf("S,S: y^t lower(A) y = \n");
	start[0] = 0; end[0] = num_vec_y;
	start[1] = 0; end[1] = num_vec_y;
	qAp = inner_prod; ldQAP = end[0]-start[0];
	ops->MultiVecQtAP('S','S',multi_vec[1],mat,multi_vec[1],0, 
		start,end,qAp,ldQAP,multi_vec[2],ops);
	for (row = 0; row < end[0]-start[0]; ++row) {
		for (col = 0; col < end[1]-start[1]; ++col) {
			ops->Printf("%6.4e\t",qAp[row+col*ldQAP]);
		}
		ops->Printf("\n");
	}
	ops->Printf("S,D: y^t lower(A) y = \n");
	start[0] = 1; end[0] = num_vec_y;
	start[1] = 1; end[1] = num_vec_y;
	qAp = inner_prod; ldQAP = end[0]-start[0];
	ops->MultiVecQtAP('S','D',multi_vec[1],mat,multi_vec[1],0, 
		start,end,qAp,1,multi_vec[2],ops);
	for (row = 0; row < end[0]-start[0]; ++row) {
		ops->Printf("%6.4e\t",qAp[row]);	
	}
	ops->Printf("\n");
	ops->Printf("S,N: y^t lower(A) y = \n");
	start[0] = 1; end[0] = num_vec_y;
	start[1] = 1; end[1] = num_vec_y;
	qAp = inner_prod; ldQAP = end[0]-start[0];
	ops->MultiVecQtAP('S','N',multi_vec[1],mat,multi_vec[1],0, 
		start,end,qAp,ldQAP,multi_vec[2],ops);
	for (row = 0; row < end[0]-start[0]; ++row) {
		for (col = 0; col < end[1]-start[1]; ++col) {
			ops->Printf("%6.4e\t",qAp[row+col*ldQAP]);
		}
		ops->Printf("\n");
	}		
	ops->Printf("N,N: y^t A y = \n");
	start[0] = 0; end[0] = num_vec_y;
	start[1] = 0; end[1] = num_vec_y;
	qAp = inner_prod; ldQAP = end[0]-start[0];
	ops->MultiVecQtAP('N','N',multi_vec[1],mat,multi_vec[1],0, 
		start,end,qAp,ldQAP,multi_vec[2],ops);
	for (row = 0; row < end[0]-start[0]; ++row) {
		for (col = 0; col < end[1]-start[1]; ++col) {
			ops->Printf("%6.4e\t",qAp[row+col*ldQAP]);
		}
		ops->Printf("\n");
	}	
	ops->Printf("N,N: x^t A y = \n");
	start[0] = 1; end[0] = num_vec_x;
	start[1] = 1; end[1] = num_vec_y;
	qAp = inner_prod; ldQAP = end[0]-start[0];
	ops->MultiVecQtAP('N','N',multi_vec[0],mat,multi_vec[1],0, 
		start,end,qAp,ldQAP,multi_vec[2],ops);
	for (row = 0; row < end[0]-start[0]; ++row) {
		for (col = 0; col < end[1]-start[1]; ++col) {
			ops->Printf("%6.4e\t",qAp[row+col*ldQAP]);
		}
		ops->Printf("\n");
	}
	
	ops->Printf("y = x coef + y beta\n");
	start[0] = 0; end[0] = num_vec_x;
	start[1] = 0; end[1] = num_vec_y;
	int ldc = end[0]-start[0], incb = 1;
	double *coef  = inner_prod; 
	double *gamma = coef+ldc*(end[1]-start[1]);
	for (row = 0; row < end[1]-start[1]; ++row) {
		gamma[row] = row; 
	}
	ops->MultiVecLinearComb(multi_vec[0],multi_vec[1],0,start,end,
		coef,ldc,gamma,incb,ops);
	ops->MultiVecView(multi_vec[1],start[1],end[1],ops);
	ops->Printf("coef = \n");
	for (row = 0; row < end[0]-start[0]; ++row) {
		for (col = 0; col < end[1]-start[1]; ++col) {
			ops->Printf("%6.4e\t",coef[row+col*ldc]);
		}
		ops->Printf("\n");
	}
	ops->Printf("beta = \n");
	for (row = 0; row < end[1]-start[1]; ++row) {
		ops->Printf("%6.4e\t",gamma[row*incb]);
	}
	ops->Printf("\n");

	ops->MultiVecDestroy(&(multi_vec[0]),num_vec_x,ops);
	ops->MultiVecDestroy(&(multi_vec[1]),num_vec_y,ops);
	ops->MultiVecDestroy(&(multi_vec[2]),num_vec_x+num_vec_y,ops);
	return 0;
}
