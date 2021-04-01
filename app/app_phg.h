/**
 *    @file  app_phg.h
 *   @brief  app of phg 
 *
 *  phg�Ĳ����ӿ�, ��� VEC 
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

#if OPS_USE_PHG

#include	<phg.h>

void OPS_PHG_Set (struct OPS_ *ops);

#endif

#endif  /* -- #ifndef _APP_PHG_H_ -- */
