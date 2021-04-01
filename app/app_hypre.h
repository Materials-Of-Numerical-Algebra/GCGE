/**
 *    @file  app_hypre.h
 *   @brief  app of hypre 
 *
 *  hypre的操作接口, 针对 VEC 
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/12/13
 *      Revision:  none
 */
#ifndef  _APP_HYPRE_H_
#define  _APP_HYPRE_H_

#include	"ops.h"
#include	"app_lapack.h"

#if OPS_USE_HYPRE

#include "_hypre_utilities.h"
#include "_hypre_parcsr_ls.h"
#include "HYPRE_krylov.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE.h"

void OPS_HYPRE_Set (struct OPS_ *ops);

#endif

#endif  /* -- #ifndef _APP_HYPRE_H_ -- */
