#include "Reductor.hpp"
#include <cstdio>

__int64 _start, _freq, _end;

int main()
{
	Reductor* reductor;
	try
	{
		reductor = new Reductor();
	}
	catch (ERROR_& e)
	{
		fprintf(stderr, "%s\n", e.msg);
	}


	reductor->makeTestSet(128 * 1024 * 1024);

	//reductor->getCPU()->test(cpu_simple, 100);
	reductor->getCPU()->test(cpu_reduction, 1);
	reductor->getCPU()->test(cpu_kahansum, 1);

	reductor->getGPU()->allocBuffer();

	for (int j = 32; j <= 256; j *= 2)
	{
		printf("== work_size : %d == ", j);
		reductor->getGPU()->test_1d(gpu_reduction_global, 1, j);
		reductor->getGPU()->test_1d(gpu_reduction_local, 1, j);
	}
		
	for(int i=16;i<=32;i*=2)
		for (int j = 16; j <= 32; j *= 2)
		{
			printf("== work_size : %d %d == ", i, j);
			reductor->getGPU()->test_2d(gpu_reduction_global, 1, 1024, i, j);
			reductor->getGPU()->test_2d(gpu_reduction_local, 1, 1024, i, j);
		}

	reductor->getGPU()->removeBuffer();

	reductor->removeTestSet();
	
	delete reductor;

	return 0;
}