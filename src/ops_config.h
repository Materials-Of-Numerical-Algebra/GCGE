/**
 *    @file  ops_config.h
 *   @brief  配置文件
 *
 *  配置文件
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/9/13
 *      Revision:  none
 */
#ifndef  _OPS_CONFIG_H_
#define  _OPS_CONFIG_H_

/* 当OPS_USE_SLEPC或PHG或HYPRE为1时, 需要让OPS_USE_MPI为1 */
/* TODO 当 OPS_INTEL_MKL 为 1 时, CCS 接口需要加速 矩阵乘以向量 */
#define  OPS_USE_HYPRE     0
#define  OPS_USE_INTEL_MKL 1 
#define  OPS_USE_MATLAB    0
#define  OPS_USE_MEMWATCH  0
#define  OPS_USE_MPI       0 
#define  OPS_USE_MUMPS     0 
#define  OPS_USE_OMP       1
#define  OPS_USE_PHG       1
#define  OPS_USE_PETSC     0 
#define  OPS_USE_SLEPC     0 
#define  OPS_USE_UMFPACK   0
/* 表示只打印0进程的输出信息 */
#define  PRINT_RANK    0

/* MATLAB 中 blas lapack 库的函数名不加 _ */
#if OPS_USE_MATLAB
#define FORTRAN_WRAPPER(x) x
#else
#define FORTRAN_WRAPPER(x) x ## _
#endif

#if OPS_USE_OMP
#define OMP_NUM_THREADS 8
#endif

#if OPS_USE_MEMWATCH
#include "../test/memwatch.h"
#endif


#endif  /* -- #ifndef _OPS_CONFIG_H_ -- */
