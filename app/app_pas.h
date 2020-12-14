/**
 *    @file  app_pas.h
 *   @brief  PAS 特征值求解器 
 *
 *  PAS 特征值求解器 
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/29
 *      Revision:  none
 */
#ifndef  _APP_PAS_H_
#define  _APP_PAS_H_

#include    "ops.h"
#include    "app_lapack.h"

typedef struct PASMAT_ {
	void *XX  ; /* LAPACKMAT */
	void ***QX;
	void **QQ ; void **P;
	int  num_levels; int level_aux;
} PASMAT;
typedef struct PASVEC_ {
	void **x ; /* LAPACKVEC */
	void ***q;
	void **P ; 
	int  num_levels; int level_aux;
} PASVEC;

void OPS_PAS_Set (struct OPS_ *ops, struct OPS_ *app_ops);

#endif   /* -- #ifndef _APP_PAS_H_ -- */
