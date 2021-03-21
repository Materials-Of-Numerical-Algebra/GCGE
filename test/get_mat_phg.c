/**
 *    @file  get_mat_phg.c
 *   @brief   
 *
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/9/13
 *      Revision:  none
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "ops.h"

#if USE_PHG 
#include <phg.h>

#if (PHG_VERSION_MAJOR <= 0 && PHG_VERSION_MINOR < 9)
# undef ELEMENT
typedef SIMPLEX ELEMENT;
#endif

static void
build_matrices(MAT *matA, MAT *matB, DOF *u_h)
{
    int N = u_h->type->nbas;	/* number of basis functions in an element */
    int i, j;
    GRID *g = u_h->g;
    ELEMENT *e;
    FLOAT A[N][N], B[N][N], vol;
    static FLOAT *B0 = NULL;
    int I[N];

    if (B0 == NULL && g->nroot > 0) {
	/* (\int \phi_j\cdot\phi_i)/vol is independent of element */
	FreeAtExit(B0);
	B0 = phgAlloc(N * N * sizeof(*B0));
	e = g->roots;
	vol = phgGeomGetVolume(g, e);
	for (i = 0; i < N; i++)
	    for (j = 0; j <= i; j++)
		B0[i * N + j] = B0[i + j * N] =
		    phgQuadBasDotBas(e, u_h, j, u_h, i, QUAD_DEFAULT) / vol;
    }

    assert(u_h->dim == 1);
    ForAllElements(g, e) {
	vol = phgGeomGetVolume(g, e);
	for (i = 0; i < N; i++) {
	    I[i] = phgMapE2L(matA->cmap, 0, e, i);
	    for (j = 0; j <= i; j++) {
		/* \int \grad\phi_j\cdot\grad\phi_i */
		A[j][i] = A[i][j] =
		    phgQuadGradBasDotGradBas(e, u_h, j, u_h, i, QUAD_DEFAULT);
		/* \int \phi_j\cdot\phi_i */
		B[j][i] = B[i][j] = B0[i * N + j] * vol;
	    }
	}

	/* loop on basis functions */
	for (i = 0; i < N; i++) {
	    if (phgDofDirichletBC(u_h, e, i, NULL, NULL, NULL, DOF_PROJ_NONE))
		continue;
	    phgMatAddEntries(matA, 1, I + i, N, I, A[i]); 
	    phgMatAddEntries(matB, 1, I + i, N, I, B[i]); 
	}
    }
}

static int
bc_map(int bctype)
{
    return DIRICHLET;	/* set Dirichlet BC on all boundaries */
}

int
CreateMatrixPHG(void **matA, void **matB, void **dofU, void **mapM, void **gridG, int argc, char *argv[])
{
#if 1
	/* Register GCGE options before phgInit */
	static int int_tmp; static double dbl_tmp; static char *str_tmp; static BOOLEAN bln_tmp;
	phgOptionsRegisterString("-eps_type", "str", &str_tmp);
	phgOptionsRegisterInt   ("-eps_nev" , "int", &int_tmp);
	phgOptionsRegisterInt   ("-eps_ncv" , "int", &int_tmp);
	phgOptionsRegisterInt   ("-eps_mpd" , "int", &int_tmp);
	phgOptionsRegisterString("-eps_max_it"  , "int", &int_tmp);
	phgOptionsRegisterFloat ("-eps_tol"     , "dbl", &dbl_tmp);
	phgOptionsRegisterNoArg ("-eps_conv_abs", "bln", &bln_tmp);	
	phgOptionsRegisterNoArg ("-eps_conv_rel", "bln", &bln_tmp);	

	phgOptionsRegisterNoArg ("-eps_monitor"     , "bln", &bln_tmp);	
	phgOptionsRegisterNoArg ("-eps_monitor_conv", "bln", &bln_tmp);	
	
	phgOptionsRegisterInt   ("-eps_lobpcg_blocksize", "int", &int_tmp);
	phgOptionsRegisterFloat ("-eps_lobpcg_restart"  , "dbl", &dbl_tmp);	
		
   	phgOptionsRegisterInt   ("-gcge_max_niter", "int", &int_tmp);
   	phgOptionsRegisterFloat ("-gcge_abs_tol"  , "dbl", &dbl_tmp);
   	phgOptionsRegisterFloat ("-gcge_rel_tol"  , "dbl", &dbl_tmp);
   	phgOptionsRegisterInt   ("-gcge_compW_cg_max_iter", "int", &int_tmp);
   	phgOptionsRegisterString("-gcge_initX_orth_method", "str", &str_tmp);
   	phgOptionsRegisterString("-gcge_compP_orth_method", "str", &str_tmp);
   	phgOptionsRegisterString("-gcge_compW_orth_method", "str", &str_tmp);

	phgOptionsRegisterInt("-nevConv"  , "int", &int_tmp);
	phgOptionsRegisterInt("-nevMax"   , "int", &int_tmp);
	phgOptionsRegisterInt("-blockSize", "int", &int_tmp);
	phgOptionsRegisterInt("-nevInit"  , "int", &int_tmp);
	
	phgOptionsRegisterInt("-use_slepc_eps", "int", &int_tmp);
#endif

    static char *fn = "../data/cube4.dat";
    static int mem_max = 3000;
    //size_t mem, mem_peak;
    int i;
    //int pre_refines = 13;
    int pre_refines = 3;
    GRID *g;
    DOF *u_h;
    MAP *map;
    MAT *A, *B;
    double wtime;

    //phgOptionsPreset("-dof_type P3");
    phgOptionsPreset("-dof_type P3");

    phgOptionsRegisterFilename("-mesh_file", "Mesh filename", &fn);
    phgOptionsRegisterInt("-pre_refines", "Pre-refines", &pre_refines);
    phgOptionsRegisterInt("-mem_max", "Maximum memory (MB)", &mem_max);

    phgInit(&argc, &argv);
    phgPrintf( "CreateMatrixPHG\n" );

    if (DOF_DEFAULT->mass_lumping == NULL)
	phgPrintf("Order of FE bases: %d\n", DOF_DEFAULT->order);
    else
	phgPrintf("Order of FE bases: %d\n", DOF_DEFAULT->mass_lumping->order0);

    g = phgNewGrid(-1);
    phgImportSetBdryMapFunc(bc_map);
    /*phgSetPeriodicity(g, X_MASK | Y_MASK | Z_MASK);*/
    if (!phgImport(g, fn, FALSE))
	phgError(1, "can't read file \"%s\".\n", fn);
    /* pre-refinement */
    phgPrintf("Now begin phgRefineAllElements...\n");
    phgRefineAllElements(g, pre_refines);
    phgPrintf("phgRefineAllElements finished\n");

    u_h = phgDofNew(g, DOF_DEFAULT, 1, "u_h", DofInterpolation);
    phgPrintf("after phgDofNew\n");
#if 0
    /* All-Neumann BC */
    phgDofSetDirichletBoundaryMask(u_h, 0);
#else
    /* All-Dirichlet BC */
    phgDofSetDirichletBoundaryMask(u_h, BDRY_MASK);
    phgPrintf("after phgDofSetDirichletBoundaryMask\n");
#endif
    /* set random initial values for the eigenvectors */
    phgDofRandomize(u_h, i == 0 ? 123 : 0);
    phgPrintf("after phgDofRandomize\n");

    phgPrintf("\n");
    if (phgBalanceGrid(g, 1.2, -1, NULL, 0.))
       phgPrintf("Repartition mesh\n");
    phgPrintf("%"dFMT" DOF, %"dFMT
	  " elements, %d submesh%s, load imbalance: %lg\n",
	  DofGetDataCountGlobal(u_h), g->nleaf_global,
	  g->nprocs, g->nprocs > 1 ? "es" : "", (double)g->lif);
    wtime = phgGetTime(NULL);
    map = phgMapCreate(u_h, NULL);
    {
       MAP *m = phgMapRemoveBoundaryEntries(map);
       A = phgMapCreateMat(m, m);
       B = phgMapCreateMat(m, m);
       phgMapDestroy(&m);
    }
    build_matrices(A, B, u_h);
    phgPrintf("Build matrices, matrix size: %"dFMT", wtime: %0.2lfs\n",
	  A->rmap->nglobal, phgGetTime(NULL) - wtime);
    wtime = phgGetTime(NULL);

    if (!A->assembled)
	    phgMatAssemble(A);
    phgMatPack(A);
    if (!B->assembled)
	    phgMatAssemble(B);
    phgMatPack(A);

    *matA  = (void *)A;
    *matB  = (void *)B;
    *dofU  = (void *)u_h;
    *mapM  = (void *)map;
    *gridG = (void *)g;

    phgPrintf( "CreateMatrixPHG\n" );
    //phgFinalize();
    return 0;
}

int
DestroyMatrixPHG(void **matA, void **matB, void **dofU, void **mapM, void **gridG, int argc, char *argv[])
{
    phgPrintf( "DestroyMatrixPHG\n" );
    //phgInit(&argc, &argv);

    phgMatDestroy((MAT**) matA);
    phgMatDestroy((MAT**) matB);

    phgDofFree   ((DOF**) dofU);
    phgMapDestroy((MAP**) mapM);
    phgFreeGrid  ((GRID**)gridG);

    phgPrintf( "DestroyMatrixPHG\n" );
    //phgFinalize();
    return 0;
}

#endif
