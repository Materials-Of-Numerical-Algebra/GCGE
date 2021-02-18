/**
 *    @file  app_phg.h
 *   @brief  app of phg 
 *
 *  phg的操作接口, 针对 VEC 
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/12/13
 *      Revision:  none
 */
#ifndef  _APP_PHG_H_
#define  _APP_PHG_H_

#include	"ops.h"
#include	"app_lapack.h"

#if USE_PHG

#include	<phg.h>

void OPS_PHG_Set (struct OPS_ *ops);

#endif

#endif  /* -- #ifndef _APP_PHG_H_ -- */
