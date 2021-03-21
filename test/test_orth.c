/**
 *    @file  orth.c
 *   @brief  orthonormalization 
 *
 *  Õý½»»¯²âÊÔ
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/14
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include	<memory.h>

#include    "ops.h"
#include    "ops_orth.h"
#include    "app_lapack.h"

int TestOrth(void *mat, struct OPS_ *ops) 
{
	double dense_mat[1024], inner_prod[1024], *dbl_ws;
	
	int nrows, ncols, ldm, start_dm, end_dm, length, int_ws[256];
	int start[2], end[2], row, col, idx;
	void **multi_vec[2] = {NULL,NULL}, *B;
	
	memset(dense_mat,0,1024*sizeof(double));
	memset(inner_prod,0,1024*sizeof(double));
	memset(int_ws,0,256*sizeof(int));
	
	srand(0);

	ops->Printf("Modified Gram-Schmidt\n");
	ops->Printf("mat B:\n");
	//ops->MatView(mat,ops);
	int num_vec = 10;
	for (idx = 0; idx < 2; ++idx) {
		ops->MultiVecCreateByMat(&(multi_vec[idx]),num_vec,mat,ops);
		ops->MultiVecSetRandomValue(multi_vec[idx],0,num_vec,ops);
	}
	
	start[0] = 0; end[0] = 5;
	start[1] = 5; end[1] = 10;
	ops->MultiVecAxpby(1.0,multi_vec[0],0.0,multi_vec[0],start,end,ops);
	//((LAPACKVEC*)multi_vec[0])->data[0] = 1.0;
	//((LAPACKVEC*)multi_vec[0])->data[1] = 0.0;
	//((LAPACKVEC*)multi_vec[0])->data[2] = 0.0;
	ops->Printf("multi vec x:\n");
	//ops->MultiVecView(multi_vec[0],0,num_vec,ops);
	int block_size = 8;
	//MultiVecOrthSetup_ModifiedGramSchmidt(block_size,2,1e-8,
	//		multi_vec[1],dense_mat,ops);
	MultiVecOrthSetup_BinaryGramSchmidt(block_size,5,1e-8,
			multi_vec[1],dense_mat,ops);
	start[0] = 0; end[0] = num_vec; B = mat;
	ops->Printf("start = %d, end = %d\n",start[0],end[0]);
	ops->MultiVecOrth(multi_vec[0],start[0],&(end[0]),B,ops);
	//ops->MultiVecOrth(multi_vec[0],start[0],&(end[0]),B,ops);
	ops->Printf("start = %d, end = %d\n",start[0],end[0]);
	ops->Printf("multi vec x (orth):\n");
	//ops->MultiVecView(multi_vec[0],0,num_vec,ops);
	ops->Printf("x^t B x = \n");
	start[0] = 0; end[0] = num_vec;
	start[1] = 0; end[1] = num_vec;
	ops->MultiVecQtAP('S','N',multi_vec[0],B,multi_vec[0],0,
		start,end,inner_prod,end[0]-start[0],multi_vec[1],ops);
	for (row = 0; row < end[0]-start[0]; ++row) {
		for (col = 0; col < end[1]-start[1]; ++col) {
			ops->Printf("%6.4e\t",inner_prod[row+col*(end[0]-start[0])]);
		}
		ops->Printf("\n");
	}
	
	start[0] = 2; end[0] = num_vec; B = mat;
	ops->Printf("start = %d, end = %d\n",start[0],end[0]);
	ops->MultiVecOrth(multi_vec[0],start[0],&(end[0]),B,ops);
	ops->Printf("start = %d, end = %d\n",start[0],end[0]);
	ops->Printf("multi vec x (orth):\n");
	ops->MultiVecView(multi_vec[0],0,num_vec,ops);
	ops->Printf("x^t B x = \n");
	start[0] = 0; end[0] = num_vec;
	start[1] = 0; end[1] = num_vec;
	ops->MultiVecQtAP('S','N',multi_vec[0],B,multi_vec[0],0,
		start,end,inner_prod,end[0]-start[0],multi_vec[1],ops);
	for (row = 0; row < end[0]-start[0]; ++row) {
		for (col = 0; col < end[1]-start[1]; ++col) {
			ops->Printf("%6.4e\t",inner_prod[row+col*(end[0]-start[0])]);
		}
		ops->Printf("\n");
	}
	
	start[0] = 2; end[0] = num_vec; B = mat;
	ops->MultiVecSetRandomValue(multi_vec[0],start[0],end[0],ops);
	ops->Printf("start = %d, end = %d\n",start[0],end[0]);
	ops->MultiVecOrth(multi_vec[0],start[0],&(end[0]),B,ops);
	ops->Printf("start = %d, end = %d\n",start[0],end[0]);
	ops->Printf("multi vec x (orth):\n");
	ops->MultiVecView(multi_vec[0],0,num_vec,ops);
	ops->Printf("x^t B x = \n");
	start[0] = 0; end[0] = num_vec;
	start[1] = 0; end[1] = num_vec;
	ops->MultiVecQtAP('S','N',multi_vec[0],B,multi_vec[0],0,
		start,end,inner_prod,end[0]-start[0],multi_vec[1],ops);
	for (row = 0; row < end[0]-start[0]; ++row) {
		for (col = 0; col < end[1]-start[1]; ++col) {
			ops->Printf("%6.4e\t",inner_prod[row+col*(end[0]-start[0])]);
		}
		ops->Printf("\n");
	}
	
	
	
	
	
	
	for (idx = 0; idx < 2; ++idx) {
		ops->MultiVecDestroy(&(multi_vec[idx]),num_vec,ops);
	}

	ops->Printf("Dense Matrix QR Fraction\n");
	nrows = 10; ncols = 5; ldm = nrows;
	for (idx = 0; idx < nrows*ncols; ++idx) {
		dense_mat[idx] = ((double)rand())/((double)RAND_MAX+1);
	}
	for (idx = 0; idx < nrows; ++idx) {
		dense_mat[ldm  +idx] = dense_mat[idx];
		dense_mat[ldm*2+idx] = dense_mat[idx];
		//dense_mat[ldm*3+idx] = dense_mat[idx];
		//dense_mat[ldm*4+idx] = dense_mat[idx];
	}
	LAPACKVEC x;
	x.nrows = nrows    ; x.ncols = ncols;
	x.data  = dense_mat; x.ldd   = ldm  ;
	ops->Printf("multi vec x:\n");
	ops->lapack_ops->MultiVecView((void **)(&x),0,x.ncols,ops->lapack_ops);
	
	start_dm = 0; end_dm = ncols; ldm    = nrows;
	dbl_ws  = dense_mat+nrows*ncols ; length = 256-nrows*ncols;
	ops->Printf("start = %d, end = %d\n",start_dm,end_dm);
	ops->DenseMatOrth(dense_mat,nrows,ldm,start_dm,&end_dm,
		1e-10,dbl_ws,length,int_ws);
	ops->Printf("start = %d, end = %d\n",start_dm,end_dm);
	ops->Printf("multi vec x (orth):\n");
	ops->lapack_ops->MultiVecView((void **)(&x),0,x.ncols,ops->lapack_ops);

	start_dm = end_dm; end_dm = ncols;
	ops->Printf("start = %d, end = %d\n",start_dm,end_dm);
	ops->DenseMatOrth(dense_mat,nrows,ldm,start_dm,&end_dm,
		1e-10,dbl_ws,length,int_ws);
	ops->Printf("start = %d, end = %d\n",start_dm,end_dm);
	ops->Printf("multi vec x (orth):\n");
	ops->lapack_ops->MultiVecView((void **)(&x),0,x.ncols,ops->lapack_ops);

	start_dm = 0; end_dm = ncols;
	ops->Printf("start = %d, end = %d\n",start_dm,end_dm);
	ops->DenseMatOrth(dense_mat,nrows,ldm,start_dm,&end_dm,
		1e-10,dbl_ws,length,int_ws);
	ops->Printf("start = %d, end = %d\n",start_dm,end_dm);
	ops->Printf("multi vec x (orth):\n");
	ops->lapack_ops->MultiVecView((void **)(&x),0,x.ncols,ops->lapack_ops);

	ops->Printf("x^t x = \n");
	start[0] = 0; end[0] = ncols;
	start[1] = 0; end[1] = ncols;
	ops->lapack_ops->MultiVecInnerProd('N',(void *)(&x),(void *)(&x),0,
		start,end,inner_prod,end[0]-start[0],ops);
	for (row = 0; row < end[0]-start[0]; ++row) {
		for (col = 0; col < end[1]-start[1]; ++col) {
			ops->Printf("%6.4e\t",inner_prod[row+col*(end[0]-start[0])]);
		}
		ops->Printf("\n");
	}

	return 0;
}

