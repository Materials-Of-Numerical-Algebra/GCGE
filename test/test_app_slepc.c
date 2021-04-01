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
int TestVec              (void *mat, struct OPS_ *ops);
int TestMultiVec         (void *mat, struct OPS_ *ops);
int TestOrth             (void *mat, struct OPS_ *ops);
int TestLinearSolver     (void *mat, struct OPS_ *ops);
int TestMultiLinearSolver(void *mat, struct OPS_ *ops);
int TestEigenSolverGCG   (void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);
int TestEigenSolverPAS   (void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);
int TestMultiGrid        (void *A, void *B, struct OPS_ *ops);

/* test EPS in SLEPc */
int TestEPS(void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);

#define OPS_USE_PHG_MAT  1
#define OPS_USE_FILE_MAT 0

#if OPS_USE_SLEPC
#include <slepceps.h>

/*
  Create an application context to contain data needed by the
  application-provided call-back routines, ops->MultiLinearSolver().
*/
typedef struct {
	KSP ksp;
	Vec rhs;
	Vec sol;
} AppCtx;

static void AppCtxCreate(AppCtx *user, Mat petsc_mat)
{
	double time_start, time_end; 
	time_start = MPI_Wtime();
	KSPCreate(PETSC_COMM_WORLD,&(user->ksp));
	KSP ksp = user->ksp;
	Mat A   = petsc_mat, F;
	PC  pc;

	KSPSetOperators(ksp,A,A);
	PetscBool flg_mumps = PETSC_TRUE, flg_mumps_ch = PETSC_FALSE;
	PetscOptionsGetBool(NULL,NULL,"-use_mumps_lu",&flg_mumps,NULL);
	PetscOptionsGetBool(NULL,NULL,"-use_mumps_ch",&flg_mumps_ch,NULL);
	if (flg_mumps || flg_mumps_ch) {
		KSPSetType(ksp,KSPPREONLY);
		PetscInt  ival,icntl;
		PetscReal val;
		KSPGetPC(ksp,&pc);
		if (flg_mumps) {
			PCSetType(pc,PCLU);
		} else if (flg_mumps_ch) {
			MatSetOption(A,MAT_SPD,PETSC_TRUE); /* set MUMPS id%SYM=1 */
			PCSetType(pc,PCCHOLESKY);
		}
		PCFactorSetMatSolverType(pc,MATSOLVERMUMPS);
		PCFactorSetUpMatSolverType(pc); /* call MatGetFactor() to create F */
		PCFactorGetMatrix(pc,&F);

		/* sequential ordering */
		icntl = 7; ival = 2;
		MatMumpsSetIcntl(F,icntl,ival);

		/* threshold for row pivot detection */
		MatMumpsSetIcntl(F,24,1);
		icntl = 3; val = 1.e-6;
		MatMumpsSetCntl(F,icntl,val);

		/* compute determinant of A */
		MatMumpsSetIcntl(F,33,0);
	}
	KSPSetFromOptions(ksp);
	/* Get info from matrix factors */
	KSPSetUp(ksp);

	time_end = MPI_Wtime();
	int rank;
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	if (rank == 0) {
		printf("FACTORIZATION time %f\n", time_end-time_start);
	}

	return;
}
static void AppCtxDestroy(AppCtx *user)
{
	KSPDestroy(&(user->ksp));
	return;
}
void KSP_MultiLinearSolver(void *mat, void **b, void **x, int *start, int *end, struct OPS_ *ops)
{
	assert(end[0]-start[0] == end[1]-start[1]);
	AppCtx *user = (AppCtx*)ops->multi_linear_solver_workspace;
	int i, length = end[0] - start[0];
	for (i = 0; i < length; ++i) {
		BVGetColumn((BV)b,start[0]+i,&(user->rhs));
		BVGetColumn((BV)x,start[1]+i,&(user->sol));
		KSPSolve(user->ksp,user->rhs,user->sol);
		BVRestoreColumn((BV)b,start[0]+i,&(user->rhs));
		BVRestoreColumn((BV)x,start[1]+i,&(user->sol));
	}
	return;
}

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
	slepc_ops->Printf("%s", help);
	
	void *matA, *matB; OPS *ops;

	/* 得到PETSC矩阵A, B, 规模为n*m */
   	Mat      slepc_matA, slepc_matB;
   	PetscBool flg;
#if OPS_USE_PHG_MAT
	void *phg_mat_A, *phg_mat_B, *phg_dof_U, *phg_map_M, *phg_grid_G;
	CreateMatrixPHG (&phg_mat_A, &phg_mat_B, &phg_dof_U, &phg_map_M, &phg_grid_G, argc, argv);
	MatrixConvertPHG2PETSC((void **)(&slepc_matA), &phg_mat_A);
	MatrixConvertPHG2PETSC((void **)(&slepc_matB), &phg_mat_B);
	DestroyMatrixPHG(&phg_mat_A, &phg_mat_B, &phg_dof_U, &phg_map_M, &phg_grid_G, argc, argv);
#elif OPS_USE_FILE_MAT
	//const char filename_matA[] = "/share/home/hhxie/liyu/MatrixCollection/Andrews.petsc.bin;
	//const char filename_matA[] = "/share/home/hhxie/liyu/MatrixCollection/c-65.petsc.bin;
	//const char filename_matA[] = "/share/home/hhxie/liyu/MatrixCollection/Ga10As10H30.petsc.bin;
	//const char filename_matA[] = "/share/home/hhxie/liyu/MatrixCollection/Ga3As3H12.petsc.bin;
	//const char filename_matA[] = "/share/home/hhxie/liyu/MatrixCollection/Ga41As41H72.petsc.bin;
	//const char filename_matA[] = "/share/home/hhxie/liyu/MatrixCollection/Si5H12.petsc.bin;
	//const char filename_matA[] = "/share/home/hhxie/liyu/MatrixCollection/SiO2.petsc.bin;
	
	char filename_matA[PETSC_MAX_PATH_LEN];
	PetscOptionsGetString(NULL,NULL,"-file",filename_matA,sizeof(filename_matA),&flg);
	if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a file name with the -file option");
	slepc_ops->Printf("%s\n",filename_matA);
	
	PetscViewer    viewer;
	PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename_matA,FILE_MODE_READ,&viewer); 
	MatCreate(PETSC_COMM_WORLD,&slepc_matA); 
	MatSetFromOptions(slepc_matA);
	MatLoad(slepc_matA,viewer); 
	PetscViewerDestroy(&viewer);
#if 1
	PetscReal shift = 0.0;
	PetscOptionsGetReal(NULL,NULL,"-shift",&shift,&flg);
	int row_start, row_end, i;
	MatGetOwnershipRange(slepc_matA,&row_start,&row_end);
	for (i = row_start; i < row_end; ++i) {
	   MatSetValue(slepc_matA, i, i, shift, ADD_VALUES);
	}
	MatAssemblyBegin(slepc_matA, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(slepc_matA, MAT_FINAL_ASSEMBLY);
#endif
	int nrows, ncols;
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
   	PetscInt n = 120, m = 120;
   	GetPetscMat(&slepc_matA, &slepc_matB, n, m);
#endif

	//slepc_ops->MatView((void*)slepc_matA, slepc_ops);
	//slepc_ops->MatView((void*)slepc_matB, slepc_ops);

	//pas_ops->MatView((void*)&pas_matA,pas_ops);
	//pas_ops->MatView((void*)&pas_matB,pas_ops);

	ops = slepc_ops; matA = (void*)(slepc_matA); matB = (void*)(slepc_matB);
	
	//TestMultiVec(matA,ops);
	//TestMultiLinearSolver(matA,ops);
	//TestOrth(matA,ops);
	PetscInt use_slepc_eps = 0;
	PetscOptionsGetInt(NULL,NULL,"-use_slepc_eps",&use_slepc_eps,&flg);
	if (use_slepc_eps)
		TestEPS(matA,matB,use_slepc_eps,argc,argv,ops);
	else {
		int flag = 0;
#if OPS_USE_MUMPS
		AppCtx user; flag = 0;
		if (flag != 0) {
			AppCtxCreate(&user, (Mat)matA);
			ops->multi_linear_solver_workspace = (void*)&user;
			ops->MultiLinearSolver = KSP_MultiLinearSolver;
		}
#endif
		TestEigenSolverGCG(matA,matB,flag,argc,argv,ops);
#if OPS_USE_MUMPS
		if (flag != 0) {
			AppCtxDestroy(&user);
		}
#endif
		//TestEigenSolverGCG(matA,matB,0,argc,argv,ops);
		//TestEigenSolverPAS(matA,matB,0,argc,argv,ops);	
	}
		
	//TestMultiGrid(matA,matB,ops);

	/* 销毁petsc矩阵 */
   	MatDestroy(&slepc_matA);
   	MatDestroy(&slepc_matB);
	
	OPS_Destroy (&slepc_ops);

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


int TestEPS(void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops)
{
	EPS eps; EPSType type; 
	PetscInt nev, ncv, mpd, max_it, nconv, its;
	PetscReal tol;
	nev = 200; ncv = 2*nev; mpd = ncv;
	tol = 1e-8; max_it = 2000;
	EPSCreate(PETSC_COMM_WORLD,&eps);
	EPSSetOperators(eps,(Mat)A,(Mat)B);
	if (B==NULL)
	   EPSSetProblemType(eps,EPS_HEP);
	else 
	   EPSSetProblemType(eps,EPS_GHEP);
	switch (flag) {
		case 1:
			EPSSetType(eps,EPSLANCZOS);
			break;
		case 2:
			EPSSetType(eps,EPSKRYLOVSCHUR);
			break;
		case 3:
			EPSSetType(eps,EPSGD);
			break;
		case 4:
			EPSSetType(eps,EPSJD);
			break;
		case 5:
			EPSSetType(eps,EPSRQCG);
			break;
		case 6:
			EPSSetType(eps,EPSLOBPCG);
			break;
		default:
			EPSSetType(eps,EPSKRYLOVSCHUR);
			//EPSSetType(eps,EPSLOBPCG);
			break;
	}
	EPSSetDimensions(eps,nev,ncv,mpd);
	EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);
	EPSSetTolerances(eps, tol, max_it);
	//EPSSetConvergenceTest(eps,EPS_CONV_REL);
	EPSSetConvergenceTest(eps,EPS_CONV_ABS);

	EPSSetFromOptions(eps);
	EPSSetUp(eps);
	//EPSView(eps,PETSC_VIEWER_STDOUT_WORLD);
	double time_start, time_interval;
#if OPS_USE_MPI
	time_start = MPI_Wtime();
#else
	time_start = clock();
#endif
	EPSSolve(eps);
#if OPS_USE_MPI
	time_interval = MPI_Wtime() - time_start;
	ops->Printf("Time is %.3f\n", time_interval);
#else
	time_interval = clock()-time_start;
	ops->Printf("Time is %.3f\n", (double)(time_interval)/CLOCKS_PER_SEC);
#endif


	EPSGetType(eps,&type);
	EPSGetConverged(eps,&nconv);
	EPSGetIterationNumber(eps, &its);
	PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);
	PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);
	PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);
	PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);
#if 0
	int i; PetscScalar eigr;
	for (i = 0; i < nconv; ++i) {
		EPSGetEigenvalue(eps,i,&eigr,NULL);
		PetscPrintf(PETSC_COMM_WORLD,"%d: %6.14e\n",1+i,eigr);
	}
#else
	PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);
	//EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD);
	EPSErrorView(eps,EPS_ERROR_ABSOLUTE,PETSC_VIEWER_STDOUT_WORLD);
	EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);
	PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);
#endif
	EPSDestroy(&eps);
	return 0;
}






#endif
