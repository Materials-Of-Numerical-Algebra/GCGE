/**
 *    @file  ops_config.h
 *   @brief  �����ļ�
 *
 *  �����ļ�
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/9/13
 *      Revision:  none
 */
#ifndef  _OPS_CONFIG_H_
#define  _OPS_CONFIG_H_

/* ��USE_SLEPC��PHG��HYPREΪ1ʱ, ��Ҫ��USE_MPIΪ1 */
#define  USE_HYPRE     1
#define  USE_INTEL_MKL 1
#define  USE_MATLAB    0
#define  USE_MEMWATCH  0
#define  USE_MPI       1
#define  USE_MUMPS     1 
#define  USE_OMP       0 
#define  USE_PHG       1
#define  USE_PETSC     1
#define  USE_SLEPC     1
#define  USE_UMFPACK   0
/* ��ʾֻ��ӡ0���̵������Ϣ */
#define  PRINT_RANK    0

/* MATLAB �� blas lapack ��ĺ��������� _ */
#if USE_MATLAB
#define FORTRAN_WRAPPER(x) x
#else
#define FORTRAN_WRAPPER(x) x ## _
#endif

#if USE_OMP
#define OMP_NUM_THREADS 4
#endif

#if USE_MEMWATCH
#include "../test/memwatch.h"
#endif


#endif  /* -- #ifndef _OPS_CONFIG_H_ -- */
