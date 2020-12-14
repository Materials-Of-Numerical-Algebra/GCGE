#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "ops.h"
#include "app_slepc.h"
#include "app_pas.h"
/* run this program using the console pauser or add your own getch, system("pause") or input loop */
int TestVec              (void *mat, struct OPS_ *ops);
int TestMultiVec         (void *mat, struct OPS_ *ops);
int TestOrth             (void *mat, struct OPS_ *ops);
int TestLinearSolver     (void *mat, struct OPS_ *ops);
int TestMultiLinearSolver(void *mat, struct OPS_ *ops);
int TestEigenSolver      (void *A, void *B, int argc, char *argv[], struct OPS_ *ops);
int TestMultiGrid        (void *A, void *B, struct OPS_ *ops);

#define TEST_PAS     0
#define USE_PHG_MAT  0
#define USE_FILE_MAT 1

#if USE_SLEPC
void MatrixConvertPHG2PETSC(void **petsc_mat,  void **phg_mat);
int  CreateMatrixPHG (void **matA, void **matB, void **dofU, void **mapM, void **gridG, int argc, char *argv[]);
int  DestroyMatrixPHG(void **matA, void **matB, void **dofU, void **mapM, void **gridG, int argc, char *argv[]);

static char help[] = "Test App of SLEPC.\n";
static void GetPetscMat(Mat *A, Mat *B, PetscInt n, PetscInt m);
int TestAppSLEPC(int argc, char *argv[]) 
{
	
	SlepcInitialize(&argc,&argv,(char*)0,help);
	PetscMPIInt   rank, size;
  	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  	MPI_Comm_size(PETSC_COMM_WORLD, &size);	
	
	OPS *slepc_ops = NULL;
	OPS_Create (&slepc_ops);
	OPS_SLEPC_Set (slepc_ops);
	OPS_Setup (slepc_ops);
#if TEST_PAS	
	OPS *pas_ops = NULL;
	OPS_Create (&pas_ops);
	OPS_PAS_Set (pas_ops,slepc_ops);
	OPS_Setup (pas_ops);
#endif	
	void *matA, *matB; OPS *ops;
	int nrows, ncols, row, col;

	/* 得到PETSC矩阵A, B, 规模为n*m */
   	Mat      slepc_matA, slepc_matB;
#if USE_PHG_MAT
	void *phg_mat_A, *phg_mat_B, *phg_dof_U, *phg_map_M, *phg_grid_G;
	CreateMatrixPHG (&phg_mat_A, &phg_mat_B, &phg_dof_U, &phg_map_M, &phg_grid_G, argc, argv);
	MatrixConvertPHG2PETSC((void **)(&slepc_matA), &phg_mat_A);
	MatrixConvertPHG2PETSC((void **)(&slepc_matB), &phg_mat_B);
	DestroyMatrixPHG(&phg_mat_A, &phg_mat_B, &phg_dof_U, &phg_map_M, &phg_grid_G, argc, argv);
#elif USE_FILE_MAT
	//const char filename_matA[] = "/share/home/hhxie/MATRIX/MatrixCollection/Ga3As3H12.petsc.bin";
	//const char filename_matA[] = "/share/home/hhxie/MATRIX/fem/A_5.petsc.bin";
	//const char filename_matA[] = "/share/home/hhxie/MATRIX/MatrixCollection/Andrews.petsc.bin";
	const char filename_matA[] = "/share/home/hhxie/MATRIX/MatrixCollection/SiO2.petsc.bin";
	slepc_ops->Printf("%s\n",filename_matA);
	PetscViewer    viewer;
	PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename_matA,FILE_MODE_READ,&viewer); 
	MatCreate(PETSC_COMM_WORLD,&slepc_matA); 
	MatSetFromOptions(slepc_matA);
	MatLoad(slepc_matA,viewer); 
	PetscViewerDestroy(&viewer);
	int row_start, row_end;
	MatGetOwnershipRange(slepc_matA,&row_start,&row_end);
	for (row = row_start; row < row_end; ++row) {
	   MatSetValue(slepc_matA, row, row, 0.0, ADD_VALUES);
	}
	MatAssemblyBegin(slepc_matA, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(slepc_matA, MAT_FINAL_ASSEMBLY);
	MatGetSize(slepc_matA,&nrows,&ncols);
	slepc_ops->Printf("%d, %d\n",nrows,ncols);

#if 1
	slepc_matB = NULL;
#else 
        	
	const char filename_matB[] = "/share/home/hhxie/MATRIX/fem/M_5.petsc.bin";
	slepc_ops->Printf("%s\n",filename_matB);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename_matB,FILE_MODE_READ,&viewer); 
	MatCreate(PETSC_COMM_WORLD,&slepc_matB); 
	MatLoad(slepc_matB,viewer); 
	MatGetSize(slepc_matB,&nrows,&ncols);
	slepc_ops->Printf("%d, %d\n",nrows,ncols);
	PetscViewerDestroy(&viewer);
#endif
	
#else
   	//PetscInt n = 3750, m = 3750;
   	PetscInt n = 33, m = 33;
   	GetPetscMat(&slepc_matA, &slepc_matB, n, m);
#endif

	//slepc_ops->MatView((void*)slepc_matA, slepc_ops);
	//slepc_ops->MatView((void*)slepc_matB, slepc_ops);
#if TEST_PAS	
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
#endif
	//pas_ops->MatView((void*)&pas_matA,pas_ops);
	//pas_ops->MatView((void*)&pas_matB,pas_ops);
#if TEST_PAS
	ops = pas_ops; matA = &pas_matA; matB = &pas_matB;
#else
	ops = slepc_ops; matA = (void*)(slepc_matA); matB = (void*)(slepc_matB);
#endif
	
	//TestMultiVec(matA,ops);
	//TestMultiLinearSolver(matA,ops);
	//TestOrth(matA,ops);
	TestEigenSolver(matA,matB,argc,argv,ops);
	//TestMultiGrid(matA,matB,ops);
 #if TEST_PAS
	free(lapack_matA.data); lapack_matA.data = NULL;
	free(lapack_matB.data); lapack_matB.data = NULL;

 	free(pas_matA.QQ); pas_matA.QQ = NULL;
 	free(pas_matA.QX); pas_matA.QX = NULL;
 	free(pas_matB.QQ); pas_matB.QQ = NULL;
 	free(pas_matB.QX); pas_matB.QX = NULL;
   	BVDestroy(&slepc_vec);
 #endif	
	/* 销毁petsc矩阵 */
   	MatDestroy(&slepc_matA);
   	MatDestroy(&slepc_matB);
	
	OPS_Destroy (&slepc_ops);
#if TEST_PAS
	OPS_Destroy (&pas_ops);
#endif

	SlepcFinalize();
	
	return 0;
}

/* 创建 2-D possion 差分矩阵 A B */
static void GetPetscMat(Mat *A, Mat *B, PetscInt n, PetscInt m)
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
