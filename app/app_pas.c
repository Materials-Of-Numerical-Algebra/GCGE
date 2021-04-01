/**
 *    @file  app_pas.c
 *   @brief  PAS 接口  
 *
 *  单向量与多向量结构是统一的, 只维护多向量 
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/29
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include	<assert.h>
#include    <math.h>
#include    <memory.h>

#include    "app_pas.h"


#if OPS_USE_MPI
int       PASMAT_COMM_COLOR;
MPI_Comm  PASMAT_COMM[2]   ;
MPI_Comm  PASMAT_INTERCOMM ;
#endif



static void MatView (PASMAT *mat, struct OPS_ *ops)
{
	int level_aux = mat->level_aux;
	int ncols     = 0;
	ops->Printf("XX:\n");
	if (mat->XX != NULL) {
		ncols     = ((LAPACKMAT*)mat->XX)->ncols;
		ops->lapack_ops->MatView(mat->XX,ops->lapack_ops);
	}
	ops->Printf("QX:\n");
	if (mat->QX != NULL) {
		ops->app_ops->MultiVecView(mat->QX[level_aux],0,ncols,ops->app_ops);
	}	
	ops->Printf("QQ:\n");
	ops->app_ops->MatView(mat->QQ[level_aux],ops->app_ops);
	return;
}
/* multi-vec */
static void MultiVecCreateByMat (PASVEC **des_vec, int num_vec, PASMAT *src_mat, struct OPS_ *ops)
{
	int level_aux = src_mat->level_aux;
	(*des_vec) = malloc(sizeof(PASVEC));
	(*des_vec)->level_aux  = level_aux; 
	(*des_vec)->num_levels = src_mat->num_levels;
	(*des_vec)->q = malloc((*des_vec)->num_levels*sizeof(void**));
	ops->lapack_ops->MultiVecCreateByMat(&((*des_vec)->x),num_vec,
		src_mat->XX,ops->lapack_ops);
	ops->app_ops->MultiVecCreateByMat(&((*des_vec)->q[level_aux]),num_vec,
		src_mat->QQ[level_aux],ops->app_ops);
	return;
}
static void MultiVecCreateByMultiVec (PASVEC **des_vec, int num_vec, PASVEC *src_vec, 
		struct OPS_ *ops)
{
	int level_aux = src_vec->level_aux;
	(*des_vec) = malloc(sizeof(PASVEC));
	(*des_vec)->level_aux  = level_aux;
	(*des_vec)->num_levels = src_vec->num_levels;
	(*des_vec)->q = malloc((*des_vec)->num_levels*sizeof(void**));
	ops->lapack_ops->MultiVecCreateByMultiVec(&((*des_vec)->x),num_vec,
		src_vec->x,ops->lapack_ops);
	ops->app_ops->MultiVecCreateByMultiVec(&((*des_vec)->q[level_aux]),num_vec,
		src_vec->q[level_aux],ops->app_ops);
	return;
}
static void MultiVecDestroy (PASVEC **des_vec, int num_vec, struct OPS_ *ops)
{
	int level_aux = (*des_vec)->level_aux;
	ops->lapack_ops->MultiVecDestroy(&((*des_vec)->x),num_vec,ops->lapack_ops);
	ops->app_ops->MultiVecDestroy(&((*des_vec)->q[level_aux]),num_vec,ops->app_ops);
	free((*des_vec)->q); (*des_vec)->q = NULL;
	free(*des_vec); *des_vec = NULL;
	return;
}
static void MultiVecView (PASVEC *x, int start, int end, struct OPS_ *ops)
{
	int level_aux = x->level_aux;
	ops->Printf("x:\n");
	ops->lapack_ops->MultiVecView(x->x,start,end,ops->lapack_ops);
	ops->Printf("q:\n");
	ops->app_ops->MultiVecView(x->q[level_aux],start,end,ops->app_ops);
	return;
}
static void MultiVecLocalInnerProd (char nsdIP, 
		PASVEC *x, PASVEC *y, int is_vec, int *start, int *end, 
		double *inner_prod, int ldIP, struct OPS_ *ops)
{
	int nrows, ncols, level_aux, sizeX;
	nrows = end[0]-start[0] ; ncols = end[1]-start[1];
	level_aux = x->level_aux; sizeX = ((LAPACKVEC*)(x->x))->nrows;	
 	/* 先计算 q 部分 */
	ops->app_ops->MultiVecLocalInnerProd(nsdIP,x->q[level_aux],y->q[level_aux],is_vec,
		start,end,inner_prod,ldIP,ops->app_ops);
	double *matQ, *matP;
	matQ = ((LAPACKVEC*)(x->x))->data+sizeX*start[0];
	matP = ((LAPACKVEC*)(y->x))->data+sizeX*start[1];		
#if OPS_USE_MPI	
	int local_rank = -1;
	MPI_Comm_rank(PASMAT_COMM[0],&local_rank);
	/* 0 进程计算 x 部分 */	
	if (PASMAT_COMM_COLOR == 0 && local_rank == 0) {
		ops->DenseMatQtAP('S',nsdIP,sizeX,sizeX,nrows,ncols,
			1.0,matQ      ,sizeX, /* Q */
			    NULL      ,sizeX, /* A */
			    matP      ,sizeX, /* P */
			1.0,inner_prod,ldIP ,
			NULL);
	}
#else
	ops->DenseMatQtAP('S',nsdIP,sizeX,sizeX,nrows,ncols,
		1.0,matQ      ,sizeX, /* Q */
		    NULL      ,sizeX, /* A */
		    matP      ,sizeX, /* P */
		1.0,inner_prod,ldIP ,
		NULL);	
#endif
	return;
}
static void MultiVecInnerProd (char nsdIP, 
		PASVEC *x, PASVEC *y, int is_vec, int *start, int *end, 
		double *inner_prod, int ldIP, struct OPS_ *ops)
{
	int nrows, ncols, level_aux;
	nrows = end[0]-start[0]; ncols = end[1]-start[1];
	level_aux = x->level_aux;
	/* 先计算 q 部分 local */
	ops->app_ops->MultiVecLocalInnerProd(nsdIP,
		x->q[level_aux],y->q[level_aux],is_vec,
		start,end,inner_prod,ldIP,ops->app_ops);
	if (nsdIP == 'D') nrows = 1;	 
#if OPS_USE_MPI		
	MPI_Request request; MPI_Status status;
	/* 创建矩阵块类型, 用于MPI通讯 */
	MPI_Datatype data_type; MPI_Op op; 
	CreateMPIDataTypeSubMat(&data_type,nrows,ncols,ldIP);
	if (PASMAT_COMM_COLOR == 0) {
		CreateMPIOpSubMatSum(&op);
		/* 非阻塞求和归约, 1 个 SUBMAT_TYPE */
		MPI_Iallreduce(MPI_IN_PLACE,inner_prod,1,data_type,
			op,PASMAT_COMM[0],&request);
	}	
#endif
	int length = nrows*ncols;
	double *dbl_ws = malloc(length*sizeof(double));
	memset(dbl_ws,0,nrows*ncols*sizeof(double));
	/* x_{a_0}^{\top} y_{a_1} */
	ops->lapack_ops->MultiVecLocalInnerProd(nsdIP,x->x,y->x,is_vec,
		start,end,dbl_ws,nrows,ops->lapack_ops);	
#if OPS_USE_MPI	
	if (PASMAT_COMM_COLOR == 0) {
		/* 非阻塞完成 */
		MPI_Wait(&request, &status);
		DestroyMPIOpSubMatSum(&op);
	}
	/* 销毁矩阵块类型 */
	DestroyMPIDataTypeSubMat(&data_type);
#endif
	/* q_{a_0}^{\top}p_{a_1}+x_{a_0}^{\top}y_{a_1} */
	int inc = 1, idx;  double one = 1.0, *destin = dbl_ws;
	if (nsdIP == 'D') {
		daxpy(&ncols,&one,destin,&inc,inner_prod,&ldIP);
	}
	else {
		if (nrows == ldIP) {
			daxpy(&length,&one,destin,&inc,inner_prod,&inc);
		}
		else {
			for (idx = 0; idx < ncols; ++idx) {
				daxpy(&nrows,&one,destin,&inc,inner_prod,&inc);
				destin += nrows; inner_prod += ldIP;
			}	
		}		
	}
	free(dbl_ws);
	return;
}
static void MultiVecSetRandomValue (PASVEC *x, int start, int end, struct OPS_ *ops)
{
	int level_aux = x->level_aux;
	ops->lapack_ops->MultiVecSetRandomValue(x->x,start,end,ops->lapack_ops);
	ops->app_ops->MultiVecSetRandomValue(x->q[level_aux],start,end,ops->app_ops);
	return;
}
static void MultiVecAxpby (double alpha, PASVEC *x, 
		double beta, PASVEC *y, int *start, int *end, struct OPS_ *ops)
{
	int level_aux = y->level_aux;
	if (x==NULL) {
		ops->lapack_ops->MultiVecAxpby(alpha,NULL,beta,y->x,
			start,end,ops->lapack_ops);
		ops->app_ops->MultiVecAxpby(alpha,NULL,beta,y->q[level_aux],
			start,end,ops->app_ops);
	}
	else {
		ops->lapack_ops->MultiVecAxpby(alpha,x->x,beta,y->x,
			start,end,ops->lapack_ops);
		ops->app_ops->MultiVecAxpby(alpha,x->q[level_aux],beta,y->q[level_aux],
			start,end,ops->app_ops);
	}
	return;
}
static void MatDotMultiVec (PASMAT *mat, PASVEC *x, 
		PASVEC *y, int *start_xy, int *end_xy, struct OPS_ *ops)
{
	int level_aux = y->level_aux, ldIP, start[2], end[2];
	if (end_xy[0]-start_xy[0]<=0||end_xy[1]-start_xy[1]<=0) return;	
	double *inner_prod;
	/* QQ q */
	ops->app_ops->MatDotMultiVec(mat->QQ[level_aux],x->q[level_aux],
		y->q[level_aux],start_xy,end_xy,ops->app_ops);
	/* XX 部分是单位矩阵, QX 部分是零矩阵 */
	if (mat->QX == NULL && mat->XX == NULL) {
		ops->lapack_ops->MultiVecAxpby(1.0,x->x,0.0,y->x,start_xy,end_xy,
			ops->lapack_ops);
		return;
	}
	/* TODO: 要么 QX 与 XX 都是 NULL, 否则都不是 NULL */
	assert(mat->QX != NULL && mat->XX != NULL);
	
	int sizeX = ((LAPACKMAT*)(mat->XX))->ncols;
	ldIP = ((LAPACKVEC*)(y->x))->ldd;
	inner_prod = ((LAPACKVEC*)(y->x))->data+ldIP*start_xy[1];
	/* XQ q */
	start[0] = 0          ; end[0] = sizeX    ;
	start[1] = start_xy[0]; end[1] = end_xy[0];
	ops->app_ops->MultiVecLocalInnerProd('N',mat->QX[level_aux],
		x->q[level_aux],0,start,end,inner_prod,ldIP,ops->app_ops);
#if OPS_USE_MPI
	int nrows, ncols;
	nrows = end[0]-start[0]; ncols = end[1]-start[1];
	MPI_Request request; MPI_Status status;
	MPI_Datatype data_type; MPI_Op op; 
	CreateMPIDataTypeSubMat(&data_type,nrows,ncols,ldIP);
	if (PASMAT_COMM_COLOR == 0) {
		CreateMPIOpSubMatSum(&op);
		/* 非阻塞求和归约, 1 个 SUBMAT_TYPE */
		MPI_Iallreduce(MPI_IN_PLACE,inner_prod,1,
		      data_type,op,PASMAT_COMM[0],&request);
	}
#endif
	/* QX x */
	start[0] = 0          ; end[0] = sizeX    ;
	start[1] = start_xy[1]; end[1] = end_xy[1];
	double *coef, beta; int ldc;
	beta = 1.0;
	ldc  = ((LAPACKVEC*)(x->x))->ldd;
	coef = ((LAPACKVEC*)(x->x))->data+ldc*start_xy[0];
	ops->app_ops->MultiVecLinearComb(mat->QX[level_aux],y->q[level_aux],
		0,start,end,coef,ldc,&beta,0,ops->app_ops);
	/* XX x */
	LAPACKVEC *tmp_vec;
	ops->lapack_ops->MultiVecCreateByMat((void***)&tmp_vec,
		end_xy[1]-start_xy[1],mat->XX,ops->lapack_ops);
	start[0] = start_xy[0]; end[0] = end_xy[0];
	start[1] = 0          ; end[1] = end_xy[1]-start_xy[1];
	ops->lapack_ops->MatDotMultiVec(mat->XX,x->x,(void**)tmp_vec,
		start,end,ops->lapack_ops);
#if OPS_USE_MPI
	if (PASMAT_COMM_COLOR == 0) {
		/* 非阻塞完成 */
		MPI_Wait(&request, &status);
		DestroyMPIOpSubMatSum(&op);
	}
	/* 销毁矩阵块类型 */
	DestroyMPIDataTypeSubMat(&data_type);
#endif
	start[0] = 0          ; end[0] = end_xy[1]-start_xy[1];
	start[1] = start_xy[1]; end[1] = end_xy[1];
	ops->lapack_ops->MultiVecAxpby(1.0,(void*)tmp_vec,1.0,y->x,
		start,end,ops->lapack_ops);
	ops->lapack_ops->MultiVecDestroy((void***)&tmp_vec,end_xy[1]-start_xy[1],
		ops->lapack_ops);
	return;
}
static void MatTransDotMultiVec (PASMAT *mat, PASVEC *x, 
		PASVEC *y, int *start, int *end, struct OPS_ *ops)
{
	MatDotMultiVec (mat, x, y, start, end, ops);
	return;
}
static void MultiVecLinearComb (
		PASVEC *x    , PASVEC *y  , int is_vec, 
		int    *start, int    *end, 
		double *coef , int    ldc , 
		double *beta , int    incb, struct OPS_ *ops)
{
	int level_aux = y->level_aux;
	if (x==NULL) {
		ops->lapack_ops->MultiVecLinearComb(NULL,y->x,
			is_vec,start,end,coef,ldc,beta,incb,ops->lapack_ops);
		ops->app_ops->MultiVecLinearComb(NULL,y->q[level_aux],
			is_vec,start,end,coef,ldc,beta,incb,ops->app_ops);
	}
	else {
		ops->lapack_ops->MultiVecLinearComb(x->x,y->x,
			is_vec,start,end,coef,ldc,beta,incb,ops->lapack_ops);
		ops->app_ops->MultiVecLinearComb(x->q[level_aux],y->q[level_aux],
			is_vec,start,end,coef,ldc,beta,incb,ops->app_ops);
	}	
	return;
}
static void MultiVecQtAP (char ntsA, char nsdQAP, 
		PASVEC *mvQ, PASMAT *matA, PASVEC *mvP, int is_vec, 
		int  *startQP, int *endQP, double *qAp, int ldQAP,
		PASVEC *mv_ws, struct OPS_ *ops)
{
	assert(matA->XX==NULL&&matA->QX==NULL);
	int level_aux = mvQ->level_aux;
	int nrows = endQP[0]-startQP[0], ncols = endQP[1]-startQP[1];
	/* q QQ p */
	ops->app_ops->MultiVecQtAP(ntsA,nsdQAP,
		mvQ->q[level_aux],matA->QQ[level_aux],mvP->q[level_aux],is_vec,
		startQP,endQP,qAp,ldQAP,
		mv_ws->q[level_aux],ops->app_ops);
	double *dataQ, *dataP; int lddQ, lddP, sizeA;
	lddQ  = ((LAPACKVEC*)mvQ->x)->ldd;
	lddP  = ((LAPACKVEC*)mvP->x)->ldd;
	dataQ = ((LAPACKVEC*)mvQ->x)->data+lddQ*startQP[0];
	dataP = ((LAPACKVEC*)mvP->x)->data+lddP*startQP[1];
	sizeA = ((LAPACKVEC*)mvQ->x)->nrows;
	ops->lapack_ops->DenseMatQtAP('S',nsdQAP,sizeA,sizeA,nrows,ncols,
		1.0,dataQ,lddQ , /* Q */
			NULL ,-1   , /* A */
			dataP,lddP , /* P */
		1.0,qAp  ,ldQAP,
		NULL);
	return;
}




/* Encapsulation */
static void PAS_MatView (void *mat, struct OPS_ *ops)
{
	MatView ((PASMAT *)mat, ops);
	return;
}
/* multi-vec */
static void PAS_MultiVecCreateByMat (void ***des_vec, int num_vec, void *src_mat, struct OPS_ *ops)
{
	MultiVecCreateByMat ((PASVEC **)des_vec, num_vec, (PASMAT *)src_mat, ops);		
	return;
}
static void PAS_MultiVecCreateByMultiVec (void ***des_vec, int num_vec, void **src_vec, struct OPS_ *ops)
{
	MultiVecCreateByMultiVec ((PASVEC **)des_vec, num_vec, (PASVEC *)src_vec, ops);
	return;
}
static void PAS_MultiVecDestroy (void ***des_vec, int num_vec, struct OPS_ *ops)
{
	MultiVecDestroy ((PASVEC **)des_vec, num_vec, ops);
	return;
}
static void PAS_MultiVecView (void **x, int start, int end, struct OPS_ *ops)
{
	MultiVecView ((PASVEC *)x, start, end, ops);
	return;
}
static void PAS_MultiVecLocalInnerProd (char nsdIP, 
		void **x, void **y, int is_vec, int *start, int *end, 
		double *inner_prod, int ldIP, struct OPS_ *ops)
{
	MultiVecLocalInnerProd (nsdIP, 
			(PASVEC *)x, (PASVEC *)y, is_vec, start, end, 
			inner_prod, ldIP, ops);
	return;
}
static void PAS_MultiVecInnerProd (char nsdIP, 
		void **x, void **y, int is_vec, int *start, int *end, 
		double *inner_prod, int ldIP, struct OPS_ *ops)
{
	MultiVecInnerProd (nsdIP, 
			(PASVEC *)x, (PASVEC *)y, is_vec, start, end, 
			inner_prod, ldIP, ops);	
#if OPS_USE_MPI
	/* 创建矩阵块类型, 用于MPI通讯 */
	int nrows = end[0]-start[0], ncols = end[1]-start[1];
	if (nsdIP=='D') nrows = 1; 
	MPI_Datatype data_type;
	CreateMPIDataTypeSubMat(&data_type,nrows,ncols,ldIP);
	/* 组间通讯, 将内积广播给 1 组 */
	if (PASMAT_COMM_COLOR==0&&PASMAT_INTERCOMM!=MPI_COMM_NULL) {
		int local_rank, root_proc;
		MPI_Comm_rank(PASMAT_COMM[0],&local_rank);
		if (local_rank == 0) root_proc = MPI_ROOT;
		else root_proc = MPI_PROC_NULL;
		/* 广播 1 个 SUBMAT_TYPE */
		MPI_Bcast(inner_prod,1,data_type,root_proc,
			PASMAT_INTERCOMM);
	}
	if (PASMAT_COMM_COLOR == 1) {
		int remote_root_proc = 0; /* 发送组在组内的根进程号 */
		MPI_Bcast(inner_prod,1,data_type,remote_root_proc,
			PASMAT_INTERCOMM);
	}
	/* 销毁矩阵块类型 */
	DestroyMPIDataTypeSubMat(&data_type);
#endif
	return;
}
static void PAS_MultiVecSetRandomValue (void **x, int start, int end, struct OPS_ *ops)
{
	MultiVecSetRandomValue ((PASVEC *)x, start, end, ops);
	return;
}
static void PAS_MultiVecAxpby (double alpha, void **x, 
		double beta, void **y, int *start, int *end, struct OPS_ *ops)
{
	MultiVecAxpby (alpha, (PASVEC *)x, beta, (PASVEC *)y, start, end, ops);
	return;
}
static void PAS_MatDotMultiVec (void *mat, void **x, 
		void **y, int *start, int *end, struct OPS_ *ops)
{
	MatDotMultiVec ((PASMAT *)mat, (PASVEC *)x, 
			(PASVEC *)y, start, end, ops);
#if OPS_USE_MPI
	/* 创建矩阵块类型, 用于MPI通讯 */
	int nrows, ncols, ldd;
	LAPACKVEC *lapack_vec = (LAPACKVEC*)((PASVEC *)y)->x;
	nrows = lapack_vec->nrows; ncols = end[1]-start[1];
	ldd = lapack_vec->ldd;
	/* TODO: 如果 nrows==ldd 则数据本身是连续的,
	 *       不知道MPI对连续数据的情况是否有优化 */ 
	double *destin = lapack_vec->data+ldd*start[1];
	MPI_Datatype data_type;
	CreateMPIDataTypeSubMat(&data_type,nrows,ncols,ldd);
	/* 组间通讯, 将内积广播给 1 组 */
	if (PASMAT_COMM_COLOR==0&&PASMAT_INTERCOMM!=MPI_COMM_NULL) {
		int local_rank, root_proc;
		MPI_Comm_rank(PASMAT_COMM[0],&local_rank);
		if (local_rank == 0) root_proc = MPI_ROOT;
		else root_proc = MPI_PROC_NULL;
		/* 广播 1 个 SUBMAT_TYPE */
		MPI_Bcast(destin,1,data_type,root_proc,
			PASMAT_INTERCOMM);
	}
	if (PASMAT_COMM_COLOR == 1) {
		int remote_root_proc = 0; /* 发送组在组内的根进程号 */
		MPI_Bcast(destin,1,data_type,remote_root_proc,
			PASMAT_INTERCOMM);
	}
	/* 销毁矩阵块类型 */
	DestroyMPIDataTypeSubMat(&data_type);
#endif
	return;
}
static void PAS_MatTransDotMultiVec (void *mat, void **x, 
		void **y, int *start, int *end, struct OPS_ *ops)
{
	MatTransDotMultiVec ((PASMAT *)mat, (PASVEC *)x, 
			(PASVEC *)y, start, end, ops);
	return;
}
static void PAS_MultiVecLinearComb (
		void **x , void **y, int is_vec, 
		int    *start, int  *end, 
		double *coef , int  ldc , 
		double *beta , int  incb, struct OPS_ *ops)
{
	MultiVecLinearComb (
			(PASVEC *)x, (PASVEC *)y, is_vec, 
			start, end , 
			coef , ldc , 
			beta , incb, ops);
	return;
}
static void PAS_MultiVecQtAP (char ntsA, char nsdQAP, 
		void **mvQ , void *matA, void   **mvP, int is_vec, 
		int  *start, int  *end , double *qAp , int ldQAP ,
		void **mv_ws, struct OPS_ *ops)
{
   	if (matA==NULL) {
	   DefaultMultiVecQtAP (ntsA,nsdQAP,mvQ,matA,mvP,is_vec, 
		 start,end,qAp,ldQAP,mv_ws,ops);
   	}
   	else {
	   /* 若XX为单位矩阵, QX为零矩阵, 做特殊处理 */
	   if (((PASMAT*)matA)->XX==NULL&&((PASMAT*)matA)->QX==NULL) {
	      /* q QQ p */
	      /* y    x */
	      MultiVecQtAP (ntsA,nsdQAP,(PASVEC *)mvQ,(PASMAT *)matA,(PASVEC *)mvP,is_vec, 
		    start,end,qAp,ldQAP,(PASVEC *)mv_ws,ops);
	   }
	   else {
	      DefaultMultiVecQtAP (ntsA,nsdQAP,mvQ,matA,mvP,is_vec, 
		    start,end,qAp,ldQAP,mv_ws,ops);
	   }
   	}
  	   
#if OPS_USE_MPI
	/* 创建矩阵块类型, 用于MPI通讯 */
	int nrows = end[0]-start[0], ncols = end[1]-start[1];
	if (nsdQAP=='D') nrows = 1; 
	MPI_Datatype data_type;
	CreateMPIDataTypeSubMat(&data_type,nrows,ncols,ldQAP);
	/* 组间通讯, 将内积广播给 1 组 */
	if (PASMAT_COMM_COLOR==0&&PASMAT_INTERCOMM!=MPI_COMM_NULL) {
		int local_rank, root_proc;
		MPI_Comm_rank(PASMAT_COMM[0],&local_rank);
		if (local_rank == 0) root_proc = MPI_ROOT;
		else root_proc = MPI_PROC_NULL;
		/* 广播 1 个 SUBMAT_TYPE */
		MPI_Bcast(qAp,1,data_type,root_proc,
			PASMAT_INTERCOMM);
	}
	if (PASMAT_COMM_COLOR == 1) {
		int remote_root_proc = 0; /* 发送组在组内的根进程号 */
		MPI_Bcast(qAp,1,data_type,remote_root_proc,
			PASMAT_INTERCOMM);
	}
	/* 销毁矩阵块类型 */
	DestroyMPIDataTypeSubMat(&data_type);
#endif
	return;
}

void OPS_PAS_Set (struct OPS_ *ops, struct OPS_ *app_ops)
{
	ops->MatView                  = PAS_MatView;
	/* multi-vec */
	ops->MultiVecCreateByMat      = PAS_MultiVecCreateByMat     ;
	ops->MultiVecCreateByMultiVec = PAS_MultiVecCreateByMultiVec;
	ops->MultiVecDestroy          = PAS_MultiVecDestroy         ;
	ops->MultiVecView             = PAS_MultiVecView            ;
	ops->MultiVecLocalInnerProd   = PAS_MultiVecLocalInnerProd  ;
	ops->MultiVecInnerProd        = PAS_MultiVecInnerProd       ;
	ops->MultiVecSetRandomValue   = PAS_MultiVecSetRandomValue  ;
	ops->MultiVecAxpby            = PAS_MultiVecAxpby           ;
	ops->MultiVecLinearComb       = PAS_MultiVecLinearComb      ;
	ops->MatDotMultiVec           = PAS_MatDotMultiVec          ;
	ops->MatTransDotMultiVec      = PAS_MatTransDotMultiVec     ;
	ops->MultiVecQtAP             = PAS_MultiVecQtAP            ;

	/* dense mat */
	//ops->lapack_ops               = app_ops->lapack_ops  ;
	//ops->DenseMatQtAP             = app_ops->DenseMatQtAP;
	//ops->DenseMatOrth             = app_ops->DenseMatOrth;
	ops->app_ops                  = app_ops;

#if OPS_USE_MPI
	PASMAT_COMM_COLOR = 0;
	PASMAT_COMM[0]    = MPI_COMM_WORLD;
	PASMAT_COMM[1]    = MPI_COMM_NULL;
	PASMAT_INTERCOMM  = MPI_COMM_NULL;
#endif

	return;
}



