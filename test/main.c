/**
 *    @file  main.c
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
#include <math.h>
#include "ops.h"

int TestAppCCS   (int argc, char *argv[]);
int TestAppLAPACK(int argc, char *argv[]);
int TestAppHYPRE (int argc, char *argv[]);
int TestAppPHG   (int argc, char *argv[]);
int TestAppSLEPC (int argc, char *argv[]);

int main(int argc, char *argv[]) 
{
#if USE_MEMWATCH 
   mwStatistics( 2 );
#endif

   //TestAppLAPACK(argc, argv);
   TestAppCCS(argc, argv);
   //TestAppHYPRE(argc, argv);
   //TestAppPHG(argc, argv);
   //TestAppSLEPC(argc, argv);
   return 0;
}
