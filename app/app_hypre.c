/**
 *    @file  app_hypre.c
 *   @brief  app of hypre  
 *
 *  单向量与多向量结构是统一的
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
 
#include	"app_hypre.h"


#if USE_HYPRE

#ifdef DEBUG
#undef DEBUG
#endif

#define DEBUG 0

static void VECGetArray(hypre_ParVector *x, double **x_array)
{
   assert(x_array!=NULL);
   *x_array = x->local_vector->data;
   return;
}
static void VECRestoreArray(hypre_ParVector *x, double **x_array)
{
   assert(x_array!=NULL);
   x->local_vector->data = *x_array;
   *x_array = NULL;
   return;
}
static void VECGetSizes(hypre_ParVector *x, int *local_size, int *global_size, int *ncols)
{
   if (local_size !=NULL) *local_size  = x->local_vector->size;
   if (global_size!=NULL) *global_size = x->global_size;
   if (ncols      !=NULL) *ncols       = x->local_vector->num_vectors;
   return;
}

static void MatView (hypre_ParCSRMatrix *mat, struct OPS_ *ops)
{
   ops->Printf("hypre_ParCSRMatrixPrint\n");
   hypre_ParCSRMatrixPrint(mat, "hypre_ParCSRMatrix");
   return;
}
/* multi-vec */
static void MultiVecCreateByMat (hypre_ParVector **des_vec, int num_vec, hypre_ParCSRMatrix *src_mat, struct OPS_ *ops)
{
   MPI_Comm           comm         = hypre_ParCSRMatrixComm(src_mat);
   HYPRE_BigInt       global_size  = hypre_ParCSRMatrixGlobalNumRows(src_mat);
   HYPRE_BigInt      *partitioning = NULL;
#ifdef HYPRE_NO_GLOBAL_PARTITION
   partitioning = hypre_CTAlloc(HYPRE_Int, 2, HYPRE_MEMORY_HOST);
   hypre_ParCSRMatrixGetLocalRange(src_mat, partitioning, partitioning+1, partitioning, partitioning+1);
   partitioning[1] += 1;
#else
   HYPRE_ParCSRMatrixGetRowPartitioning(src_mat, &partitioning);
#endif
   ops->Printf ( "vec by mat partitioning: %d, %d, global_size = %d\n", partitioning[0], partitioning[1], global_size );
   *des_vec = hypre_ParMultiVectorCreate(comm, global_size, partitioning, num_vec);
   hypre_ParVectorInitialize(*des_vec);
   hypre_ParVectorSetPartitioningOwner(*des_vec, 1);
   hypre_ParVectorSetConstantValues(*des_vec, 0);
   return;
}
static void MultiVecDestroy (hypre_ParVector **des_vec, int num_vec, struct OPS_ *ops)
{
   hypre_ParVectorDestroy(*des_vec);
   *des_vec = NULL;
   return;
}
static void MultiVecView (hypre_ParVector *x, int start, int end, struct OPS_ *ops)
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
      hypre_ParVector *x, hypre_ParVector *y, int is_vec, int *start, int *end, 
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
static void MultiVecSetRandomValue (hypre_ParVector *x, int start, int end, struct OPS_ *ops)
{
   assert(start>=0&&end<=x->local_vector->num_vectors);
   int     nvec = x->local_vector->num_vectors;
   double *data = x->local_vector->data;
   x->local_vector->num_vectors = end-start;
   x->local_vector->data += start*x->local_vector->size;
   hypre_ParVectorSetRandomValues(x, rand());
   x->local_vector->num_vectors = nvec;
   x->local_vector->data        = data;
   return;
}
static void MultiVecAxpby (double alpha, hypre_ParVector *x, 
      double beta, hypre_ParVector *y, int *start, int *end, struct OPS_ *ops)
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
static void MatDotMultiVec (hypre_ParCSRMatrix *mat, hypre_ParVector *x, 
      hypre_ParVector *y, int *start, int *end, struct OPS_ *ops)
{
   assert(end[0]-start[0]==end[1]-start[1]);
   if (end[0]-start[0] == 0) return;
   int i;
   int     nvec_x = x->local_vector->num_vectors,  nvec_y = y->local_vector->num_vectors;
   double *data_x = x->local_vector->data       , *data_y = y->local_vector->data;
if (0) {
   x->local_vector->num_vectors = 1; y->local_vector->num_vectors = 1;
   x->local_vector->data += start[0]*x->local_vector->size;
   y->local_vector->data += start[1]*y->local_vector->size;
   for (i = 0; i < end[0]-start[0]; ++i) {
      hypre_ParCSRMatrixMatvec(1.0, mat, x, 0.0, y);
      x->local_vector->data += x->local_vector->size;
      y->local_vector->data += y->local_vector->size;
   }
}
else {
   /* support multi vec */
   x->local_vector->num_vectors = end[0]-start[0]; y->local_vector->num_vectors = end[1]-start[1];
   x->local_vector->data += start[0]*x->local_vector->size;
   y->local_vector->data += start[1]*y->local_vector->size;
   hypre_ParCSRMatrixMatvec(1.0, mat, x, 0.0, y);
}

   x->local_vector->num_vectors = nvec_x; y->local_vector->num_vectors = nvec_y;
   x->local_vector->data        = data_x; y->local_vector->data        = data_y;
   return;
}
static void MatTransDotMultiVec (hypre_ParCSRMatrix *mat, hypre_ParVector *x, 
      hypre_ParVector *y, int *start, int *end, struct OPS_ *ops)
{
   assert(end[0]-start[0]==end[1]-start[1]);
   if (end[0]-start[0] == 0) return;
   int i;
   int     nvec_x = x->local_vector->num_vectors,  nvec_y = y->local_vector->num_vectors;
   double *data_x = x->local_vector->data       , *data_y = y->local_vector->data;
   x->local_vector->num_vectors = end[0]-start[0]; y->local_vector->num_vectors = end[1]-start[1];
   x->local_vector->data += start[0]*x->local_vector->size;
   y->local_vector->data += start[1]*y->local_vector->size;
   hypre_ParCSRMatrixMatvecT(1.0, mat, x, 0.0, y);
   x->local_vector->num_vectors = nvec_x; y->local_vector->num_vectors = nvec_y;
   x->local_vector->data        = data_x; y->local_vector->data        = data_y;
   return;
}
static void MultiVecLinearComb (hypre_ParVector *x, hypre_ParVector *y, int is_vec, 
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
static void HYPRE_MatView (void *mat, struct OPS_ *ops)
{
   MatView((hypre_ParCSRMatrix*)mat,ops);
   return;
}
/* multi-vec */
static void HYPRE_MultiVecCreateByMat (void ***des_vec, int num_vec, void *src_mat, struct OPS_ *ops)
{
   MultiVecCreateByMat ((hypre_ParVector**)des_vec,num_vec,(hypre_ParCSRMatrix*)src_mat,ops);		
   return;
}
static void HYPRE_MultiVecDestroy (void ***des_vec, int num_vec, struct OPS_ *ops)
{
   MultiVecDestroy ((hypre_ParVector**)des_vec,num_vec,ops);
   return;
}
static void HYPRE_MultiVecView (void **x, int start, int end, struct OPS_ *ops)
{
   MultiVecView ((hypre_ParVector*)x,start,end,ops);
   return;
}
static void HYPRE_MultiVecLocalInnerProd (char nsdIP, 
      void **x, void **y, int is_vec, int *start, int *end, 
      double *inner_prod, int ldIP, struct OPS_ *ops)
{
   MultiVecLocalInnerProd (nsdIP, 
	 (hypre_ParVector*)x,(hypre_ParVector*)y,is_vec,start,end, 
	 inner_prod,ldIP,ops);
   return;
}
static void HYPRE_MultiVecSetRandomValue (void **x, int start, int end, struct OPS_ *ops)
{
   MultiVecSetRandomValue ((hypre_ParVector*)x,start,end,ops);
   return;
}
static void HYPRE_MultiVecAxpby (double alpha, void **x, 
      double beta, void **y, int *start, int *end, struct OPS_ *ops)
{
   MultiVecAxpby (alpha,(hypre_ParVector*)x,beta,(hypre_ParVector*)y,start,end,ops);
   return;
}
static void HYPRE_MatDotMultiVec (void *mat, void **x, 
      void **y, int *start, int *end, struct OPS_ *ops)
{
   MatDotMultiVec ((hypre_ParCSRMatrix*)mat,(hypre_ParVector*)x,(hypre_ParVector*)y,start,end,ops);
   return;
}
static void HYPRE_MatTransDotMultiVec (void *mat, void **x, 
      void **y, int *start, int *end, struct OPS_ *ops)
{
   MatTransDotMultiVec ((hypre_ParCSRMatrix*)mat,(hypre_ParVector*)x,(hypre_ParVector*)y,start,end,ops);
   return;
}
static void HYPRE_MultiVecLinearComb (
      void **x , void **y, int is_vec, 
      int    *start, int  *end, 
      double *coef , int  ldc , 
      double *beta , int  incb, struct OPS_ *ops)
{
   MultiVecLinearComb (
	 (hypre_ParVector*)x, (hypre_ParVector*)y, is_vec, 
	 start, end , 
	 coef , ldc , 
	 beta , incb, ops);
   return;
}

static void HYPRE_MultiGridCreate(void ***A_array, void ***B_array, void ***P_array, int *num_levels, void *A, void *B, struct OPS_ *ops)
{
    /* Create solver */
    HYPRE_Solver      amg;
    HYPRE_Int level;
    HYPRE_BoomerAMGCreate(&amg);
    /* Set some parameters (See Reference Manual for more parameters) */
    HYPRE_BoomerAMGSetPrintLevel (amg, 1);         /* print solve info + parameters */
    HYPRE_BoomerAMGSetInterpType (amg, 0);
    HYPRE_BoomerAMGSetPMaxElmts  (amg, 0);
    HYPRE_BoomerAMGSetCoarsenType(amg, 6);
    HYPRE_BoomerAMGSetMaxLevels  (amg, *num_levels);  /* maximum number of levels */

    /* Create a HYPRE_ParVector by HYPRE_ParCSRMatrix */
    hypre_ParVector    *hypre_par_vec;
    ops->MultiVecCreateByMat((void***)&hypre_par_vec, 1, A, ops);

    /* Now setup AMG */
    HYPRE_BoomerAMGSetup(amg, (hypre_ParCSRMatrix*)A, hypre_par_vec, hypre_par_vec);
    ops->MultiVecDestroy((void***)&hypre_par_vec, 1, ops);

    hypre_ParAMGData* amg_data = (hypre_ParAMGData*) amg;

    /* Get num_levels */
    *num_levels = hypre_ParAMGDataNumLevels(amg_data);
    /* Create HYPRE Matrix */
    HYPRE_ParCSRMatrix *hypre_A_array, *hypre_B_array, *hypre_P_array;
    hypre_A_array = malloc(sizeof(HYPRE_ParCSRMatrix)*(*num_levels));
    hypre_P_array = malloc(sizeof(HYPRE_ParCSRMatrix)*(*num_levels-1));

    hypre_A_array[0] = hypre_ParAMGDataAArray(amg_data)[0];
    for (level = 1; level < *num_levels; ++level)
    {
       hypre_A_array[level] = hypre_ParAMGDataAArray(amg_data)[level];
       ops->Printf("A[%d]: %10d, %10d\n", level, hypre_A_array[level]->global_num_rows, hypre_A_array[level]->global_num_cols);
       /* 这样赋值之后，当amg Destroy的时候，不会将A_array释放 */
       hypre_ParAMGDataAArray(amg_data)[level] = NULL;
    }
    for (level = 0; level < *num_levels-1; ++level)
    {
       hypre_P_array[level] = hypre_ParAMGDataPArray(amg_data)[level];
       ops->Printf("P[%d]: %10d, %10d\n", level, hypre_P_array[level]->global_num_rows, hypre_P_array[level]->global_num_cols);
       /* 这样赋值之后，当amg Destroy的时候，不会将P_array释放 */
       hypre_ParAMGDataPArray(amg_data)[level] = NULL;
    }

    /* line 2495 in src/parcsr_ls/par_amg_setup.c 
     * HYPRE在这里有相当多的情况，这里只取了一个对简单的情况 
     * 
     * hypre_ParAMGDataBlockMode(amg_data) 需要是 0 即, 不是block
     * 这个与grid_relax_type有关 */
    HYPRE_Int       num_procs;
    HYPRE_Int keepTranspose = hypre_ParAMGDataKeepTranspose(amg_data);
    hypre_MPI_Comm_size( hypre_ParCSRMatrixComm((HYPRE_ParCSRMatrix)(A)), &num_procs);
    /* B0  P0^T B0 P0  P1^T B1 P1   P2^T B2 P2 */
    hypre_ParCSRMatrix  *B_H;
    if (B!=NULL)
    {
       hypre_B_array = malloc(sizeof(hypre_ParCSRMatrix*)*(*num_levels));
       hypre_B_array[0] = (hypre_ParCSRMatrix*)B;
       for ( level = 1; level < (*num_levels); ++level )
       {
	  /* Compute standard Galerkin coarse-grid product */
	  if (hypre_ParAMGDataModularizedMatMat(amg_data))
	  {
	     B_H = hypre_ParCSRMatrixRAPKT(hypre_P_array[level-1], hypre_B_array[level-1],
		   hypre_P_array[level-1], keepTranspose);
	  }
	  else
	  {
	     hypre_BoomerAMGBuildCoarseOperatorKT(hypre_P_array[level-1], hypre_B_array[level-1] ,
		   hypre_P_array[level-1], keepTranspose, &B_H);
	  }
	  /* dropping in B_H */
	  hypre_ParCSRMatrixDropSmallEntries(B_H, hypre_ParAMGDataADropTol(amg_data),
		hypre_ParAMGDataADropType(amg_data));
	  /* if CommPkg for B_H was not built */
	  //MPI_Comm  comm = hypre_ParCSRMatrixComm((HYPRE_ParCSRMatrix)(A));
	  if (num_procs > 1 && hypre_ParCSRMatrixCommPkg(B_H) == NULL)
	  {
	     hypre_MatvecCommPkgCreate(B_H);
	  }
	  if (hypre_ParAMGDataADropTol(amg_data) <= 0.0)
	  {
	     hypre_ParCSRMatrixSetNumNonzeros(B_H);
	     hypre_ParCSRMatrixSetDNumNonzeros(B_H);
	  }
	  /* RowStarts may belong to A_array */
	  hypre_ParCSRMatrixOwnsRowStarts(B_H) = 0;
	  hypre_B_array[level] = B_H;
       }
       (*B_array) = (void**)hypre_B_array;
    }
    /* HYPRE_BoomerAMGDestroy(amg); 对于释放amg时，对AArray和PArray的操作 */
    /* if (hypre_ParAMGDataAArray(amg_data)[i])
       hypre_ParCSRMatrixDestroy(hypre_ParAMGDataAArray(amg_data)[i]);

       if (hypre_ParAMGDataPArray(amg_data)[i-1])
       hypre_ParCSRMatrixDestroy(hypre_ParAMGDataPArray(amg_data)[i-1]); */    
    (*A_array) = (void**)hypre_A_array;
    (*P_array) = (void**)hypre_P_array;
    HYPRE_BoomerAMGDestroy(amg);
    return;
}
static void HYPRE_MultiGridDestroy(void ***A_array, void ***B_array, void ***P_array, int *num_levels, struct OPS_ *ops)
{
    HYPRE_ParCSRMatrix *hypre_A_array, *hypre_B_array, *hypre_P_array;
    int level; 
    hypre_A_array = (hypre_ParCSRMatrix**)(*A_array);
    hypre_P_array = (hypre_ParCSRMatrix**)(*P_array);

    for ( level = 1; level < (*num_levels); ++level )
    {
       hypre_ParCSRMatrixDestroy(hypre_A_array[level]);
    }
    for ( level = 0; level < (*num_levels) - 1; ++level )
    {
       hypre_ParCSRMatrixDestroy(hypre_P_array[level]);
    }
    free(hypre_A_array);
    free(hypre_P_array);
    (*A_array) = NULL;
    (*P_array) = NULL;

    if (B_array!=NULL)
    {
       hypre_B_array = (hypre_ParCSRMatrix**)(*B_array);
       for ( level = 1; level < (*num_levels); ++level )
       {
	  hypre_ParCSRMatrixDestroy(hypre_B_array[level]);
       }
       free(hypre_B_array);
       (*B_array) = NULL;
    }

    return;
}

void OPS_HYPRE_Set (struct OPS_ *ops)
{
   ops->MatView                = HYPRE_MatView;
   /* multi-vec */
   ops->MultiVecCreateByMat    = HYPRE_MultiVecCreateByMat   ;
   ops->MultiVecDestroy        = HYPRE_MultiVecDestroy       ;
   ops->MultiVecView           = HYPRE_MultiVecView          ;
   ops->MultiVecLocalInnerProd = HYPRE_MultiVecLocalInnerProd;
   ops->MultiVecSetRandomValue = HYPRE_MultiVecSetRandomValue;
   ops->MultiVecAxpby          = HYPRE_MultiVecAxpby         ;
   ops->MatDotMultiVec         = HYPRE_MatDotMultiVec        ;
   ops->MatTransDotMultiVec    = HYPRE_MatTransDotMultiVec   ;
   ops->MultiVecLinearComb     = HYPRE_MultiVecLinearComb    ;
   /* multi grid */
   ops->MultiGridCreate        = HYPRE_MultiGridCreate ;
   ops->MultiGridDestroy       = HYPRE_MultiGridDestroy;
   return;
}

#endif
