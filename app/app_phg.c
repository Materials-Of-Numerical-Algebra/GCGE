/**
 *    @file  app_phg.c
 *   @brief  app of phg  
 *
 *  µ¥ÏòÁ¿Óë¶àÏòÁ¿½á¹¹ÊÇÍ³Ò»µÄ
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/12/13
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include	<assert.h>
#include	<math.h>
#include	<memory.h>
 
#include	"app_phg.h"


#if OPS_USE_PHG

#ifdef DEBUG
#undef DEBUG
#endif

#define DEBUG 0

static void VECGetArray(VEC *x, double **x_array)
{
	assert(x_array!=NULL);
	*x_array = x->data;
	return;
}
static void VECRestoreArray(VEC *x, double **x_array)
{
	assert(x_array!=NULL);
	x->data  = *x_array;
	*x_array = NULL;
	return;
}
static void VECGetSizes(VEC *x, int *local_size, int *global_size, int *ncols)
{
	if (local_size !=NULL) *local_size  = x->map->nlocal;
	if (global_size!=NULL) *global_size = x->map->nglobal;
	if (ncols      !=NULL) *ncols       = x->nvec;
	return;
}

static void phgMatDotMultiVecLocal (MAT *A, VEC *x, VEC *y)
{
	assert(A->type != PHG_DESTROYED);
	if (!A->assembled)
		phgMatAssemble(A);
	phgMatPack(A);

	int nrows, ncols, *rowsSE, row;		
	nrows = A->rmap->nlocal;
	ncols = A->cmap->nlocal;
	assert(x->map->nlocal==ncols);
	assert(y->map->nlocal==nrows);
	/* rows start and end for local data 
	 * the type of A->packed_ind[0] is size_t */
	rowsSE = malloc((nrows+1)*sizeof(int));
	for (row = 0; row < nrows+1; ++row) {
		rowsSE[row] = (int)A->packed_ind[row];
	}
#if DEBUG
	int idx, nvec, my_rank;
	my_rank = A->cmap->rank;
	if (my_rank==PRINT_RANK) {
		printf("[%d]: nrows = %d, ncols = %d, nnz_d = %d, nnz_o = %d\n", 
				my_rank, nrows, ncols, A->nnz_d, A->nnz_o); 
		//printf("size_t = %d, MKL_INT = %d, int = %d\n", sizeof(size_t),sizeof(MKL_INT),sizeof(int));
		for (row = 0; row < nrows+1; ++row) {
			printf("[%d]: rowsSE %d\n", my_rank, rowsSE[row]); 
		}
		for (row = 0; row < nrows; ++row) {
			printf("[%d]: row = %d, SE = %d,%d, nnz = %d\n",
					my_rank, row, rowsSE[row], rowsSE[row+1],rowsSE[row+1]-rowsSE[row]); 
			for (idx = rowsSE[row]; idx < rowsSE[row+1]; ++idx) {
				printf("%.4e (%d)\t", A->packed_data[idx], A->packed_cols[idx]);
			}
			printf("\n"); 
		}
	}
#endif

#if OPS_USE_INTEL_MKL
	//mkl_set_num_threads_local(MKL_NUM_THREADS);
	//#pragma omp parallel num_threads(MKL_NUM_THREADS)
	//{
	//	int id = omp_get_thread_num();
	//	printf("%d thread\n",id);
	//}

	sparse_matrix_t csrA;
	struct matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;
	/*
	 * sparse_status_t mkl_sparse_d_create_csr (
	 *       sparse_matrix_t *A,  
	 *       const sparse_index_base_t indexing,  
	 *       const MKL_INT rows,  const MKL_INT cols,  
	 *       MKL_INT *rows_start,  MKL_INT *rows_end,  MKL_INT *col_indx,  double *values);
	 * sparse_status_t mkl_sparse_destroy (sparse_matrix_t A);
	 * sparse_status_t mkl_sparse_d_mm (
	 *       const sparse_operation_t operation,  
	 *       const double alpha,  
	 *       const sparse_matrix_t A,  const struct matrix_descr descr,  const sparse_layout_t layout,  
	 *       const double *B,  const MKL_INT columns,  const MKL_INT ldb,  
	 *       const double beta,  double *C,  const MKL_INT ldc);
	 */

	/* in process */
	mkl_sparse_d_create_csr (
			&csrA,
			SPARSE_INDEX_BASE_ZERO,  
			nrows, ncols,  
			rowsSE, rowsSE+1, A->packed_cols, A->packed_data);
#if OPS_USE_OMP
	#pragma omp parallel num_threads(OMP_NUM_THREADS)
	{
		int id, length, offset;
		id     = omp_get_thread_num();
		length = x->nvec/OMP_NUM_THREADS;
		offset = length*id;
		if (id < x->nvec%OMP_NUM_THREADS) {
			++length; offset += id;
		}
		else {
			offset += x->nvec%OMP_NUM_THREADS;
		} 
		mkl_sparse_d_mm (
				SPARSE_OPERATION_NON_TRANSPOSE,
				1.0,
				csrA, descr, SPARSE_LAYOUT_COLUMN_MAJOR,  
				     x->data+offset*ncols, length, ncols,  
				0.0, y->data+offset*nrows, nrows);
	}
#else
	mkl_sparse_d_mm (
			SPARSE_OPERATION_NON_TRANSPOSE,
			1.0,
			csrA, descr, SPARSE_LAYOUT_COLUMN_MAJOR,  
			     x->data, x->nvec, ncols,  
			0.0, y->data, nrows);
#endif
	mkl_sparse_destroy (csrA);
#else

#if 0
	int *cols = A->packed_cols, i;
	memset(y->data,0,nrows*x->nvec*sizeof(double));
#if OPS_USE_OMP
	#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
	for (i = 0; i < x->nvec; ++i) {
	   double *dm, *dy, *dx;
	   dm = A->packed_data;
	   dy = y->data+nrows*i;
	   dx = x->data+ncols*i;
	   int j, k;
	   for (k = 0; k < nrows; ++k) {
	      for (j = rowsSE[k]; j < rowsSE[k+1]; ++j) {
		 dy[k] += (*dm++)*dx[cols[j]];
	      }
	   }
	}
#else
	memset(y->data,0,nrows*x->nvec*sizeof(double));
	/* PetscErrorCode MatMatMultNumericAdd_SeqAIJ_SeqDense(Mat A,Mat B,Mat C) */
	double r1,r2,r3,r4,*c1,*c2,*c3,*c4,aatmp;
	const double *aa,*b1,*b2,*b3,*b4,*av;
	const int    *aj;
	int          cn=x->nvec,bm=ncols,am=nrows;
	int          am4=4*am,bm4=4*bm,col,i,j,n,ajtmp;
	av = A->packed_data;
	b1 = x->data; b2 = b1 + bm; b3 = b2 + bm; b4 = b3 + bm;
	c1 = y->data; c2 = c1 + am; c3 = c2 + am; c4 = c3 + am;
	/* 4 cols per each iteration */
	for (col=0; col<cn-4; col += 4) {  /* over columns of C */
	   for (i=0; i<am; i++) {        /* over rows of C in those columns */
	      r1 = r2 = r3 = r4 = 0.0; 
	      n  = rowsSE[i+1] - rowsSE[i];
	      aj = A->packed_cols + rowsSE[i];
	      aa = av + rowsSE[i];
	      for (j=0; j<n; j++) {
		 aatmp = aa[j]; ajtmp = aj[j];
		 r1 += aatmp*b1[ajtmp];
		 r2 += aatmp*b2[ajtmp];
		 r3 += aatmp*b3[ajtmp];
		 r4 += aatmp*b4[ajtmp];
	      }    
	      c1[i] += r1;
	      c2[i] += r2;
	      c3[i] += r3;
	      c4[i] += r4;
	   }    
	   b1 += bm4; b2 += bm4; b3 += bm4; b4 += bm4; 
	   c1 += am4; c2 += am4; c3 += am4; c4 += am4; 
	}
	for (; col<cn; col++) {   /* over extra columns of C */
	   for (i=0; i<am; i++) {  /* over rows of C in those columns */
	      r1 = 0.0; 
	      n  = rowsSE[i+1] - rowsSE[i];
	      aj = A->packed_cols + rowsSE[i];
	      aa = av + rowsSE[i];
	      for (j=0; j<n; j++) {
		 r1 += aa[j]*b1[aj[j]];
	      }    
	      c1[i] += r1;
	   }    
	   b1 += bm;
	   c1 += am;
	}
#endif

#endif 
	free(rowsSE);
	return;
}
static void phgMatDotMultiVecRemote (MAT *A, VEC *x, VEC *y)
{
	INT i, j, n, *pc, col;
	FLOAT *x_data = NULL, *y_data = NULL, *dbl_ptr, *pd, *v, beta;
	FLOAT *offp_data = NULL, *offp_data2 = NULL;
	assert(y != NULL && x != NULL);
	if (!x->assembled)
		phgVecAssemble(x);
	if (!y->assembled)
		phgVecAssemble(y);
	assert(A->type != PHG_DESTROYED);
	if (!A->assembled)
		phgMatAssemble(A);
	phgMatPack(A);

	if (A->cmap->nprocs > 1) {
		offp_data  = phgAlloc(A->cinfo->rsize * sizeof(*offp_data ));
		if (x->nvec > 1)
			offp_data2 = phgAlloc(A->cinfo->rsize * sizeof(*offp_data2));

		x_data = x->data; y_data = y->data;
		phgMapScatterBegin(A->cinfo, 1, x_data, offp_data);
		phgMapScatterEnd  (A->cinfo, 1, x_data, offp_data);
		for (col = 1; col < x->nvec; ++col) {
			dbl_ptr = offp_data; offp_data = offp_data2; offp_data2 = dbl_ptr;
			x_data += x->map->nlocal;
			phgMapScatterBegin(A->cinfo, 1, x_data, offp_data);
			/* multiply with remote data */
			for (i = 0, v = y_data; i < A->rmap->nlocal; i++) {
				j  = A->rmap->nlocal + i;
				pc = A->packed_cols  + A->packed_ind[j];
				pd = A->packed_data  + A->packed_ind[j];
				n = (INT)(A->packed_ind[j + 1] - A->packed_ind[j]);
				if (n == 0) {
					v++;
					continue;
				}
				beta = pd[0] * offp_data2[pc[0]];
				for (j = 1; j < n; j++)
					beta += pd[j] * offp_data2[pc[j]];
				*(v++) += beta;
			}
			y_data += y->map->nlocal;
			phgMapScatterEnd(A->cinfo, 1, x_data, offp_data);
		}
		for (i = 0, v = y_data; i < A->rmap->nlocal; i++) {
			j  = A->rmap->nlocal + i;
			pc = A->packed_cols  + A->packed_ind[j];
			pd = A->packed_data  + A->packed_ind[j];
			n = (INT)(A->packed_ind[j + 1] - A->packed_ind[j]);
			if (n == 0) {
				v++;
				continue;
			}
			beta = pd[0] * offp_data[pc[0]];
			for (j = 1; j < n; j++)
				beta += pd[j] * offp_data[pc[j]];
			*(v++) += beta;
		}
		phgFree(offp_data);
		if (x->nvec > 1)
			phgFree(offp_data2);
	}
	return;
}



static void phgMatDotMultiVec (MAT *A, VEC *x, VEC *y)
{
	INT i, j, n, *pc, col;
	FLOAT *x_data = NULL, *y_data = NULL, *dbl_ptr, *pd, *v, beta;
	FLOAT *offp_data = NULL, *offp_data2 = NULL;
	assert(y != NULL && x != NULL);
	if (!x->assembled)
		phgVecAssemble(x);
	if (!y->assembled)
		phgVecAssemble(y);
	assert(A->type != PHG_DESTROYED);
	if (!A->assembled)
		phgMatAssemble(A);
	phgMatPack(A);

	if (A->cmap->nprocs > 1) {
		offp_data = phgAlloc(A->cinfo->rsize * sizeof(*offp_data ));
		x_data = x->data; y_data = y->data;
		phgMapScatterBegin(A->cinfo, 1, x_data, offp_data);
		if (x->nvec > 1)
			offp_data2 = phgAlloc(A->cinfo->rsize * sizeof(*offp_data2));
	}
	/* multiply with local data */
	phgMatDotMultiVecLocal (A, x, y);
	if (A->cmap->nprocs > 1) {
		phgMapScatterEnd  (A->cinfo, 1, x_data, offp_data);
		for (col = 1; col < x->nvec; ++col) {
			dbl_ptr = offp_data; offp_data = offp_data2; offp_data2 = dbl_ptr;
			x_data += x->map->nlocal;
			phgMapScatterBegin(A->cinfo, 1, x_data, offp_data);
			/* multiply with remote data */
			for (i = 0, v = y_data; i < A->rmap->nlocal; i++) {
				j  = A->rmap->nlocal + i;
				pc = A->packed_cols  + A->packed_ind[j];
				pd = A->packed_data  + A->packed_ind[j];
				n = (INT)(A->packed_ind[j + 1] - A->packed_ind[j]);
				if (n == 0) {
					v++;
					continue;
				}
				beta = pd[0] * offp_data2[pc[0]];
				for (j = 1; j < n; j++)
					beta += pd[j] * offp_data2[pc[j]];
				*(v++) += beta;
			}
			y_data += y->map->nlocal;
			phgMapScatterEnd(A->cinfo, 1, x_data, offp_data);
		}
		for (i = 0, v = y_data; i < A->rmap->nlocal; i++) {
			j  = A->rmap->nlocal + i;
			pc = A->packed_cols  + A->packed_ind[j];
			pd = A->packed_data  + A->packed_ind[j];
			n = (INT)(A->packed_ind[j + 1] - A->packed_ind[j]);
			if (n == 0) {
				v++;
				continue;
			}
			beta = pd[0] * offp_data[pc[0]];
			for (j = 1; j < n; j++)
				beta += pd[j] * offp_data[pc[j]];
			*(v++) += beta;
		}
		phgFree(offp_data);
		if (x->nvec > 1)
			phgFree(offp_data2);
	}
	return;
}


static void MatView (MAT *mat, struct OPS_ *ops)
{
	return;
}
/* multi-vec */
static void MultiVecCreateByMat (VEC **des_vec, int num_vec, MAT *src_mat, struct OPS_ *ops)
{
	*des_vec = phgMapCreateVec(src_mat->cmap, num_vec);
	return;
}
static void MultiVecDestroy (VEC **des_vec, int num_vec, struct OPS_ *ops)
{
	phgVecDestroy(des_vec);
	return;
}

static void MultiVecView (VEC *x, int start, int end, struct OPS_ *ops)
{
    int x_nrows, x_ncols; double *x_array;
	VECGetArray(x,&x_array);
    VECGetSizes(x,&x_nrows,NULL,&x_ncols);
    LAPACKVEC x_vec;
    x_vec.nrows = x_nrows; x_vec.ncols = x_ncols;
    x_vec.ldd   = x_nrows; x_vec.data  = x_array;
	ops->lapack_ops->MultiVecView((void**)&x_vec,start,end,ops->lapack_ops);
	return;
}
static void MultiVecLocalInnerProd (char nsdIP, 
		VEC *x, VEC *y, int is_vec, int *start, int *end, 
		double *inner_prod, int ldIP, struct OPS_ *ops)
{
	assert(is_vec==0);
	
    double *x_array, *y_array;
    int x_nrows, x_ncols, y_nrows, y_ncols;
    VECGetArray(x,&x_array); VECGetArray(y,&y_array);
    VECGetSizes(x,&x_nrows,NULL,&x_ncols);
	VECGetSizes(y,&y_nrows,NULL,&y_ncols);
	LAPACKVEC x_vec, y_vec;
	x_vec.nrows = x_nrows; y_vec.nrows = y_nrows;
	x_vec.ncols = x_ncols; y_vec.ncols = y_ncols;
	x_vec.ldd   = x_nrows; y_vec.ldd   = y_nrows;
	x_vec.data  = x_array; y_vec.data  = y_array;
	ops->lapack_ops->MultiVecLocalInnerProd(nsdIP,
			(void**)&x_vec,(void**)&y_vec,is_vec,
			start,end,inner_prod,ldIP,ops->lapack_ops);
	VECRestoreArray(x,&x_array); VECRestoreArray(y,&y_array);
	return;
}
static void MultiVecSetRandomValue (VEC *x, int start, int end, struct OPS_ *ops)
{
	int     nvec = x->nvec;
	double *data = x->data;
	x->nvec  = end-start;
	x->data += start*x->map->nlocal;
	phgVecRandomize(x, rand());
	x->nvec  = nvec;
	x->data  = data;
	return;
}
static void MultiVecAxpby (double alpha, VEC *x, 
		double beta, VEC *y, int *start, int *end, struct OPS_ *ops)
{
	assert(end[0]-start[0]==end[1]-start[1]);
	if (x==y) assert(end[0]<=start[1]||end[1]<=start[0]);

    int y_nrows, y_ncols; double *y_array;
	VECGetArray(y,&y_array);
    VECGetSizes(y,&y_nrows,NULL,&y_ncols);
    LAPACKVEC y_vec;
    y_vec.nrows = y_nrows; y_vec.ncols = y_ncols;
    y_vec.ldd   = y_nrows; y_vec.data  = y_array;
	if (x==NULL) {
		ops->lapack_ops->MultiVecAxpby(alpha,
				NULL,beta,(void**)&y_vec,start,end,ops->lapack_ops);
	}
	else {
		int x_nrows, x_ncols; double *x_array;
		VECGetArray(x,&x_array); 
		VECGetSizes(x,&x_nrows,NULL,&x_ncols);
		LAPACKVEC x_vec;
		x_vec.nrows = x_nrows; x_vec.ncols = x_ncols; 
		x_vec.ldd   = x_nrows; x_vec.data  = x_array; 
		ops->lapack_ops->MultiVecAxpby(alpha,
				(void**)&x_vec,beta,(void**)&y_vec,start,end,ops->lapack_ops);
		VECRestoreArray(x, &x_array); 
	}
	VECRestoreArray(y, &y_array);
    return;
}
static void MatDotMultiVec (MAT *mat, VEC *x, 
		VEC *y, int *start, int *end, struct OPS_ *ops)
{
	assert(end[0]-start[0]==end[1]-start[1]);
	assert(end[0]-start[0]>=0);
	if (end[0]-start[0]==0) return;
	int     nvec_x = x->nvec,  nvec_y = y->nvec;
	double *data_x = x->data, *data_y = y->data;
#if 1
	x->nvec = end[0]-start[0]; y->nvec = end[1]-start[1];
	x->data += start[0]*x->map->nlocal;
	y->data += start[1]*y->map->nlocal;
	if (0) {
	   /* y  = mat x */
	   phgMatDotMultiVecLocal (mat, x, y);
	   /* y += mat x */
	   phgMatDotMultiVecRemote(mat, x, y);
	}
	else {
	   phgMatDotMultiVec(mat, x, y);
	}
#else	
	int i;
	x->nvec = 1; y->nvec = 1;
	x->data += start[0]*x->map->nlocal;
	y->data += start[1]*y->map->nlocal;
	for (i = 0; i < end[0]-start[0]; ++i) {
		phgMatVec(MAT_OP_N, 1.0, mat, x, 0.0, &y);
		x->data += x->map->nlocal;
		y->data += y->map->nlocal;
	}
#endif
	x->nvec = nvec_x; y->nvec = nvec_y;
	x->data = data_x; y->data = data_y;
	return;
}
static void MatTransDotMultiVec (MAT *mat, VEC *x, 
		VEC *y, int *start, int *end, struct OPS_ *ops)
{
	assert(end[0]-start[0]==end[1]-start[1]);
	int     i;
	int     nvec_x = x->nvec,  nvec_y = y->nvec;
	double *data_x = x->data, *data_y = y->data;
	x->nvec = 1; y->nvec = 1;
	x->data += start[0]*x->map->nlocal;
	y->data += start[1]*y->map->nlocal;
	for (i = 0; i < end[0]-start[0]; ++i) {
		phgMatVec(MAT_OP_T, 1.0, mat, x, 0.0, &y);
		x->data += x->map->nlocal;
		y->data += y->map->nlocal;
	}
	x->nvec = nvec_x; y->nvec = nvec_y;
	x->data = data_x; y->data = data_y;
	return;
}
static void MultiVecLinearComb (VEC *x, VEC *y, int is_vec, 
		int    *start, int *end, 
		double *coef , int ldc , 
		double *beta , int incb, struct OPS_ *ops)
{
    assert(is_vec==0);
    if (x==y) assert(end[0]<=start[1]||end[1]<=start[0]);
    
    int y_nrows, y_ncols; double *y_array;
    VECGetArray(y,&y_array);
    VECGetSizes(y,&y_nrows,NULL,&y_ncols);
    LAPACKVEC y_vec;
    y_vec.nrows = y_nrows; y_vec.ncols = y_ncols;
    y_vec.ldd   = y_nrows; y_vec.data  = y_array;
    if (x==NULL) {
		ops->lapack_ops->MultiVecLinearComb(
			NULL,(void**)&y_vec,is_vec,
			start,end,coef,ldc,beta,incb,ops->lapack_ops);
    }
    else {
		int x_nrows, x_ncols;
		double *x_array;
		LAPACKVEC x_vec;
		VECGetArray(x,&x_array);
		VECGetSizes(x,&x_nrows,NULL,&x_ncols);
		x_vec.nrows = x_nrows; x_vec.ncols = x_ncols; 
		x_vec.ldd   = x_nrows; x_vec.data  = x_array; 
		ops->lapack_ops->MultiVecLinearComb(
			(void**)&x_vec,(void**)&y_vec,is_vec,
			start,end,coef,ldc,beta,incb,ops->lapack_ops);
		VECRestoreArray(x, &x_array);
    }
    VECRestoreArray(y, &y_array);
    return;
}


/* Encapsulation */
static void PHG_MatView (void *mat, struct OPS_ *ops)
{
	MatView((MAT*)mat,ops);
	return;
}
/* multi-vec */
static void PHG_MultiVecCreateByMat (void ***des_vec, int num_vec, void *src_mat, struct OPS_ *ops)
{
	MultiVecCreateByMat ((VEC**)des_vec,num_vec,(MAT*)src_mat,ops);		
	return;
}
static void PHG_MultiVecDestroy (void ***des_vec, int num_vec, struct OPS_ *ops)
{
	MultiVecDestroy ((VEC**)des_vec,num_vec,ops);
	return;
}
static void PHG_MultiVecView (void **x, int start, int end, struct OPS_ *ops)
{
	MultiVecView ((VEC*)x,start,end,ops);
	return;
}
static void PHG_MultiVecLocalInnerProd (char nsdIP, 
		void **x, void **y, int is_vec, int *start, int *end, 
		double *inner_prod, int ldIP, struct OPS_ *ops)
{
	MultiVecLocalInnerProd (nsdIP, 
			(VEC*)x,(VEC*)y,is_vec,start,end, 
			inner_prod,ldIP,ops);
	return;
}
static void PHG_MultiVecSetRandomValue (void **x, int start, int end, struct OPS_ *ops)
{
	MultiVecSetRandomValue ((VEC*)x,start,end,ops);
	return;
}
static void PHG_MultiVecAxpby (double alpha, void **x, 
		double beta, void **y, int *start, int *end, struct OPS_ *ops)
{
	MultiVecAxpby (alpha,(VEC*)x,beta,(VEC*)y,start,end,ops);
	return;
}
static void PHG_MatDotMultiVec (void *mat, void **x, 
		void **y, int *start, int *end, struct OPS_ *ops)
{
	MatDotMultiVec ((MAT*)mat,(VEC*)x,(VEC*)y,start,end,ops);
	return;
}
static void PHG_MatTransDotMultiVec (void *mat, void **x, 
		void **y, int *start, int *end, struct OPS_ *ops)
{
	MatTransDotMultiVec ((MAT*)mat,(VEC*)x,(VEC*)y,start,end,ops);
	return;
}
static void PHG_MultiVecLinearComb (
		void **x , void **y, int is_vec, 
		int    *start, int  *end, 
		double *coef , int  ldc , 
		double *beta , int  incb, struct OPS_ *ops)
{
	MultiVecLinearComb (
			(VEC*)x, (VEC*)y, is_vec, 
			start, end , 
			coef , ldc , 
			beta , incb, ops);
	return;
}



int PHG_GetOptionFromCommandLine (
		const char *name, char type, void *value,
		int argc, char* argv[], struct OPS_ *ops)
{
   static int *int_value; static double *dbl_value; static char *str_value;
   switch (type) {
	   case 'i':
		   int_value = (int*)value; 
		   phgOptionsRegisterInt(name, NULL, int_value);
		   break;
	   case 'f':
		   dbl_value = (double*)value; 
		   phgOptionsRegisterFloat(name, NULL, dbl_value);
		   break;
	   case 's':
		   str_value = (char*)value;
		   phgOptionsRegisterString(name, NULL, &str_value);
		   //DefaultGetOptionFromCommandLine(name, type, value, argc, argv, ops);
		   break;
	   default:
		   break;
   }
   return 0;
}

void OPS_PHG_Set (struct OPS_ *ops)
{
	//ops->GetOptionFromCommandLine = PHG_GetOptionFromCommandLine;
	ops->Printf                 = DefaultPrintf;
	ops->MatView                = PHG_MatView;
	/* multi-vec */
	ops->MultiVecCreateByMat    = PHG_MultiVecCreateByMat   ;
	ops->MultiVecDestroy        = PHG_MultiVecDestroy       ;
	ops->MultiVecView           = PHG_MultiVecView          ;
	ops->MultiVecLocalInnerProd = PHG_MultiVecLocalInnerProd;
	ops->MultiVecSetRandomValue = PHG_MultiVecSetRandomValue;
	ops->MultiVecAxpby          = PHG_MultiVecAxpby         ;
	ops->MatDotMultiVec         = PHG_MatDotMultiVec        ;
	ops->MatTransDotMultiVec    = PHG_MatTransDotMultiVec   ;
	ops->MultiVecLinearComb     = PHG_MultiVecLinearComb    ;
	/* multi grid */
	return;
}

#endif
