#include <stdio.h>
#include <stdlib.h>

#include "ops.h"

int TestAppLAPACK(int argc, char *argv[]);
int TestAppSLEPC (int argc, char *argv[]);

int main(int argc, char *argv[]) 
{
#if USE_MEMWATCH 
    mwStatistics( 2 );
#endif
	TestAppLAPACK(argc, argv);
	//TestAppSLEPC(argc, argv);
	return 0;
}

