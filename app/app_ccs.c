/**
 *    @file  app_ccs.c
 *   @brief  app of ccs  
 *
 *  单向量与多向量结构是统一的
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *           ZJ Wang, for OpenMP
 *
 *       Created:  2020/8/13
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include	<assert.h>
#include	<math.h>
#include	<memory.h>
 
#include	"app_ccs.h"

static void MatView (CCSMAT *mat, struct OPS_ *ops)
{
	/* 第 i_row[i] 行, 第 j 列 元素非零, i data[i]
	 * j_col[j] <= i < j_col[j+1] */
	LAPACKVEC *multi_vec;
	ops->MultiVecCreateByMat((void ***)(&multi_vec), mat->ncols, mat, ops);
	int col, i; double *destin; 
	for (col = 0; col < mat->ncols; ++col) {
		for (i = mat->j_col[col]; i < mat->j_col[col+1]; ++i) {
			destin  = multi_vec->data+(multi_vec->ldd)*col+mat->i_row[i];
			*destin = mat->data[i];
		}
	}
	ops->lapack_ops->MatView((void *)multi_vec, ops->lapack_ops);
	ops->lapack_ops->MultiVecDestroy((void ***)(&multi_vec), mat->ncols, ops->lapack_ops);
	return;
}
/* multi-vec */
static void MultiVecCreateByMat (LAPACKVEC **des_vec, int num_vec, CCSMAT *src_mat, struct OPS_ *ops)
{
	(*des_vec)        = malloc(sizeof(LAPACKVEC));
	(*des_vec)->nrows = src_mat->ncols   ; 
	(*des_vec)->ncols = num_vec          ;
	(*des_vec)->ldd   = (*des_vec)->nrows;
	(*des_vec)->data  = malloc(((*des_vec)->ldd)*((*des_vec)->ncols)*sizeof(double));
	memset((*des_vec)->data,0,((*des_vec)->ldd)*((*des_vec)->ncols)*sizeof(double));
	return;
}
static void MatDotMultiVec (CCSMAT *mat, LAPACKVEC *x, 
		LAPACKVEC *y, int *start, int *end, struct OPS_ *ops)
{
	assert(end[0]-start[0]==end[1]-start[1]);
	assert(y->nrows==y->ldd);
	assert(x->nrows==x->ldd);
	int length = end[0]-start[0]; int col; 
	if (mat!=NULL) {
		memset(y->data+(y->ldd)*start[1],0,(y->ldd)*length*sizeof(double));
#if USE_OMP
		#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
		for (col = 0; col < length; ++col) {
			int i, j;
			double *dm, *dx, *dy; int *i_row;
			dm = mat->data; i_row = mat->i_row;
			dx = x->data+(x->ldd)*(start[0]+col);
			dy = y->data+(y->ldd)*(start[1]+col);
			for (j = 0; j < mat->ncols; ++j, ++dx) {
				for (i = mat->j_col[j]; i < mat->j_col[j+1]; ++i) {
					dy[*i_row++] += (*dm++)*(*dx);
				}
			}
		}
	}
	else {
		ops->lapack_ops->MultiVecAxpby (1.0, (void **)x, 0.0, (void **)y, 
				start, end, ops->lapack_ops);
	}
	return;
}
static void MatTransDotMultiVec (CCSMAT *mat, LAPACKVEC *x, 
		LAPACKVEC *y, int *start, int *end, struct OPS_ *ops)
{
	assert(end[0]-start[0]==end[1]-start[1]);
	assert(y->nrows==y->ldd);
	assert(x->nrows==x->ldd);
	assert(mat->nrows==mat->ncols);
	MatDotMultiVec (mat, x, y, start, end, ops);
	return;
}
static void VecCreateByMat (LAPACKVEC **des_vec, CCSMAT *src_mat, struct OPS_ *ops)
{
	MultiVecCreateByMat(des_vec,1,src_mat, ops);
	return;
}
static void MatDotVec (CCSMAT *mat, LAPACKVEC *x, LAPACKVEC *y, struct OPS_ *ops)
{
	int start[2] = {0,0}, end[2] = {1,1};
	MatDotMultiVec(mat,x,y,start,end, ops);
	return;
}
static void MatTransDotVec (CCSMAT *mat, LAPACKVEC *x, LAPACKVEC *y, struct OPS_ *ops)
{
	int start[2] = {0,0}, end[2] = {1,1};
	MatTransDotMultiVec(mat,x,y,start,end, ops);
	return;
}

/* Encapsulation */
static void CCS_MatView (void *mat, struct OPS_ *ops)
{
	MatView ((CCSMAT *)mat, ops);
	return;
}
/* vec */
static void CCS_VecCreateByMat (void **des_vec, void *src_mat, struct OPS_ *ops)
{
	VecCreateByMat ((LAPACKVEC **)des_vec, (CCSMAT *)src_mat, ops);
	return;
}
static void CCS_MatDotVec (void *mat, void *x, void *y, struct OPS_ *ops)
{
	MatDotVec ((CCSMAT *)mat, (LAPACKVEC *)x, (LAPACKVEC *)y, ops);
	return;
}
static void CCS_MatTransDotVec (void *mat, void *x, void *y, struct OPS_ *ops)
{
	MatTransDotVec ((CCSMAT *)mat, (LAPACKVEC *)x, (LAPACKVEC *)y, ops);
	return;
}
/* multi-vec */
static void CCS_MultiVecCreateByMat (void ***des_vec, int num_vec, void *src_mat, struct OPS_ *ops)
{
	MultiVecCreateByMat ((LAPACKVEC **)des_vec, num_vec, (CCSMAT *)src_mat, ops);		
	return;
}

static void CCS_MatDotMultiVec (void *mat, void **x, 
		void **y, int *start, int *end, struct OPS_ *ops)
{
	MatDotMultiVec ((CCSMAT *)mat, (LAPACKVEC *)x, 
			(LAPACKVEC *)y, start, end, ops);
	return;
}
static void CCS_MatTransDotMultiVec (void *mat, void **x, 
		void **y, int *start, int *end, struct OPS_ *ops)
{
	MatTransDotMultiVec ((CCSMAT *)mat, (LAPACKVEC *)x, 
			(LAPACKVEC *)y, start, end, ops);
	return;
}

void OPS_CCS_Set (struct OPS_ *ops)
{
	assert(ops->lapack_ops==NULL);
	OPS_Create (&(ops->lapack_ops));
	OPS_LAPACK_Set (ops->lapack_ops);
	ops->Printf                   = DefaultPrintf;
	ops->GetOptionFromCommandLine = DefaultGetOptionFromCommandLine;
	ops->MatView                  = CCS_MatView;
	/* vec */
	ops->VecCreateByMat           = CCS_VecCreateByMat;
	ops->VecCreateByVec           = ops->lapack_ops->VecCreateByVec   ;
	ops->VecDestroy               = ops->lapack_ops->VecDestroy       ;
	ops->VecView                  = ops->lapack_ops->VecView          ;
	ops->VecInnerProd             = ops->lapack_ops->VecInnerProd     ;
	ops->VecLocalInnerProd        = ops->lapack_ops->VecLocalInnerProd;
	ops->VecSetRandomValue        = ops->lapack_ops->VecSetRandomValue;
	ops->VecAxpby                 = ops->lapack_ops->VecAxpby         ;
	ops->MatDotVec                = CCS_MatDotVec     ;
	ops->MatTransDotVec           = CCS_MatTransDotVec;
	/* multi-vec */
	ops->MultiVecCreateByMat      = CCS_MultiVecCreateByMat;
	ops->MultiVecCreateByVec      = ops->lapack_ops->MultiVecCreateByVec     ;
	ops->MultiVecCreateByMultiVec = ops->lapack_ops->MultiVecCreateByMultiVec;
	ops->MultiVecDestroy          = ops->lapack_ops->MultiVecDestroy         ;
	ops->GetVecFromMultiVec       = ops->lapack_ops->GetVecFromMultiVec      ;
	ops->RestoreVecForMultiVec    = ops->lapack_ops->RestoreVecForMultiVec   ;
	ops->MultiVecView             = ops->lapack_ops->MultiVecView            ;
	ops->MultiVecLocalInnerProd   = ops->lapack_ops->MultiVecLocalInnerProd  ;
	ops->MultiVecInnerProd        = ops->lapack_ops->MultiVecInnerProd       ;
	ops->MultiVecSetRandomValue   = ops->lapack_ops->MultiVecSetRandomValue  ;
	ops->MultiVecAxpby            = ops->lapack_ops->MultiVecAxpby           ;
	ops->MultiVecLinearComb       = ops->lapack_ops->MultiVecLinearComb      ;
	ops->MatDotMultiVec           = CCS_MatDotMultiVec     ;
	ops->MatTransDotMultiVec      = CCS_MatTransDotMultiVec;
	return;
}

