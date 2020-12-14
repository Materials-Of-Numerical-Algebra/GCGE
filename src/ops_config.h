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

#define  USE_MATLAB   0
#define  USE_MEMWATCH 0
#define  USE_MPI      0
#define  USE_PHG      0
/* 当USE_SLEPC为1时, 需要让USE_MPI为1 */
#define  USE_SLEPC    0
/* 表示只打印0进程的输出信息 */
#define  PRINT_RANK 0

/* MATLAB 中 blas lapack 库的函数名不加 _ */
#if USE_MATLAB
#define FORTRAN_WRAPPER(x) x
#else
#define FORTRAN_WRAPPER(x) x ## _
#endif

#if USE_MEMWATCH
#include "../test/memwatch.h"
#endif


#endif  /* -- #ifndef _OPS_CONFIG_H_ -- */
