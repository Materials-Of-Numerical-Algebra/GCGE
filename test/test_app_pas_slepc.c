/**
 *    @file  test_app_slepc.c
 *   @brief  test app of SLEPC 
 *
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/17
 *      Revision:  none
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "ops.h"
#include "app_slepc.h"
#include "app_pas.h"
/* run this program using the console pauser or add your own getch, system("pause") or input loop */
int TestMultiVec         (void *mat, struct OPS_ *ops);
int TestOrth             (void *mat, struct OPS_ *ops);
int TestMultiLinearSolver(void *mat, struct OPS_ *ops);
int TestEigenSolverGCG   (void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);

#if OPS_USE_SLEPC
#include <slepceps.h>

static char help[] = "Test App of PAS in SLEPC.\n";
static void GetPetscMat(Mat *A, Mat *B, PetscInt n, PetscInt m);
int TestAppPAS_SLEPC(int argc, char *argv[]) 
{	
	SlepcInitialize(&argc,&argv,(char*)0,help);
	PetscMPIInt   rank, size;
  	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  	MPI_Comm_size(PETSC_COMM_WORLD, &size);	
	
	OPS *slepc_ops = NULL;
	OPS_Create (&slepc_ops);
	OPS_SLEPC_Set (slepc_ops);
	OPS_Setup (slepc_ops);
	slepc_ops->Printf("%s", help);
	
	OPS *pas_ops = NULL;
	OPS_Create (&pas_ops);
	OPS_PAS_Set (pas_ops,slepc_ops);
	OPS_Setup (pas_ops);
	
	void *matA, *matB; OPS *ops;

	/* 得到PETSC矩阵A, B, 规模为n*m */
   	Mat      slepc_matA, slepc_matB;

   	//PetscInt n = 3750, m = 3750;
   	//PetscInt n = 60, m = 60;
   	PetscInt n = 6, m = 6;
   	GetPetscMat(&slepc_matA, &slepc_matB, n, m);


	//slepc_ops->MatView((void*)slepc_matA, slepc_ops);
	//slepc_ops->MatView((void*)slepc_matB, slepc_ops);

	BV slepc_vec;
	{
	   Vec vector;
	   MatCreateVecs(slepc_matA,NULL,&vector);
	   BVCreate(PETSC_COMM_WORLD, &slepc_vec);
	   BVSetType(slepc_vec,BVMAT);
	   BVSetSizesFromVec(slepc_vec,vector,n*m);
	   VecDestroy(&vector);
	}
	PetscInt local_nrows, global_nrows, ncols;
	BVGetSizes(slepc_vec,&local_nrows,&global_nrows,&ncols);
	BVSetActiveColumns(slepc_vec,0,ncols);
	BVSetRandom(slepc_vec);	
	//slepc_ops->MultiVecView((void**)slepc_vec,0,ncols,slepc_ops);

	LAPACKMAT lapack_matA; 
	lapack_matA.nrows = ncols; lapack_matA.ncols = ncols; lapack_matA.ldd = ncols;
	lapack_matA.data  = malloc(ncols*ncols*sizeof(double));
	int row, col;
	for (col = 0; col < ncols; ++col) {
		for (row = 0; row < ncols; ++row) {			
			if (row == col) lapack_matA.data[row+ncols*col] = 2.0;
			else if (row-col == 1) lapack_matA.data[row+ncols*col] = -1.0;
			else if (row-col ==-1) lapack_matA.data[row+ncols*col] = -1.0;
			else lapack_matA.data[row+ncols*col] = 0.0;
		}			
	}
	LAPACKMAT lapack_matB;
	lapack_matB.nrows = ncols; lapack_matB.ncols = ncols; lapack_matB.ldd = ncols;
	lapack_matB.data  = malloc(ncols*ncols*sizeof(double));
	for (col = 0; col < ncols; ++col) {		
		for (row = 0; row < ncols; ++row) {
			if (row == col) lapack_matB.data[row+ncols*row] = 1.0;
			else lapack_matB.data[col+ncols*row] = 0.0;
		}
	}
	
	PASMAT pas_matA, pas_matB; 
	pas_matA.level_aux = 0; pas_matA.num_levels = 1;
	pas_matA.QQ = malloc(sizeof(void*));
	pas_matA.QX = malloc(sizeof(void**)); 
	pas_matA.XX    = (void*)&lapack_matA;
	pas_matA.QQ[0] = (void*)slepc_matA;
	pas_matA.QX[0] = (void**)slepc_vec;
	pas_matB.level_aux = 0; pas_matB.num_levels = 1;
	pas_matB.QQ = malloc(sizeof(void*));
	pas_matB.QX = malloc(sizeof(void**)); 
	pas_matB.XX    = (void*)&lapack_matB;
	pas_matB.QQ[0] = (void*)slepc_matB;
	pas_matB.QX[0] = (void**)slepc_vec;

	//pas_ops->MatView((void*)&pas_matA,pas_ops);
	//pas_ops->MatView((void*)&pas_matB,pas_ops);
	ops = pas_ops; matA = &pas_matA; matB = &pas_matB;

	
	//TestMultiVec(matA,ops);
	TestMultiLinearSolver(matA,ops);
	//TestOrth(matA,ops);	
	//TestEigenSolverGCG(matA,matB,0,argc,argv,ops);

	free(lapack_matA.data); lapack_matA.data = NULL;
	free(lapack_matB.data); lapack_matB.data = NULL;

 	free(pas_matA.QQ); pas_matA.QQ = NULL;
 	free(pas_matA.QX); pas_matA.QX = NULL;
 	free(pas_matB.QQ); pas_matB.QQ = NULL;
 	free(pas_matB.QX); pas_matB.QX = NULL;
   	BVDestroy(&slepc_vec);
	
	/* 销毁petsc矩阵 */
   	MatDestroy(&slepc_matA);
   	MatDestroy(&slepc_matB);
	
	OPS_Destroy (&slepc_ops);

	OPS_Destroy (&pas_ops);

	SlepcFinalize();
	
	return 0;
}

/* 创建 2-D possion 差分矩阵 A B */
void GetPetscMat(Mat *A, Mat *B, PetscInt n, PetscInt m)
{
    assert(n==m);
    PetscInt N = n*m;
    PetscInt Istart, Iend, II, i, j;
    PetscReal h = 1.0/(n+1);
    MatCreate(PETSC_COMM_WORLD,A);
    MatSetSizes(*A,PETSC_DECIDE,PETSC_DECIDE,N,N);
	//MatSetFromOptions(*A);
    MatSetUp(*A);
    MatGetOwnershipRange(*A,&Istart,&Iend);
    for (II=Istart;II<Iend;II++) {
      	i = II/n; j = II-i*n;
      	if (i>0)   { MatSetValue(*A,II,II-n,-1.0/h,INSERT_VALUES); }
      	if (i<m-1) { MatSetValue(*A,II,II+n,-1.0/h,INSERT_VALUES); }
      	if (j>0)   { MatSetValue(*A,II,II-1,-1.0/h,INSERT_VALUES); }
     	if (j<n-1) { MatSetValue(*A,II,II+1,-1.0/h,INSERT_VALUES); }
       	MatSetValue(*A,II,II,4.0/h,INSERT_VALUES);
    }
    MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);

    MatCreate(PETSC_COMM_WORLD,B);
    MatSetSizes(*B,PETSC_DECIDE,PETSC_DECIDE,N,N);
	//MatSetFromOptions(*B);
    MatSetUp(*B);
    MatGetOwnershipRange(*B,&Istart,&Iend);
    for (II=Istart;II<Iend;II++) {
      	MatSetValue(*B,II,II,1.0*h,INSERT_VALUES);
    }
    MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);
	return;
}

#endif
