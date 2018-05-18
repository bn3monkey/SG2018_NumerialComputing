#include "Reductor_CPU.hpp"
#include "my_OpenCL_util.h"

void Reductor_CPU::set(float* input, float* output, size_t size)
{
	this->input = input;
	this->output = output;
	this->size = size;
}
void Reductor_CPU::test(enum CPU_KERNEL kernel_num, int trial)
{
	float compute_time;

	fprintf(stdout, "\n^^^ Test : general CPU computation ^^^\n");
	fprintf(stdout, "   [CPU Execution] \n");
	fprintf(stdout, "     * Trial number : %d Test Size : %d Test Kernel : %s\n", trial, size, str_kernel[(int)kernel_num]);

	CHECK_TIME_START;

	int i = trial;
	while(i--)
		func_kernel[(int)kernel_num] (input, output, size);

	CHECK_TIME_END(compute_time);

	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time/(float)trial);
	fprintf(stdout, "   [Check Results] \n");
	fprintf(stdout, "     * C[%lu] = %f\n\n", 0, output[0]);

}
//1. for 문으로 모두 더하기 수행
void CPU_simple(float* input, float* output, int size)
{
	int i;
	float sum = 0.0f;
	for (i=size-1;i>=0;i--)
	{
		sum += input[i];
	}
	*output = sum;
}
//2. reduction 수행
void CPU_reduction(float* input, float* output, int size)
{
	int i, j;
	float sum = 0.0f;
	
	memcpy(output, input, sizeof(float)*size);

	for (i = size / 2; i > 0; i >>= 1)
	{
		for (j = i - 1; j >= 0; j--)
		{
			//if (j == 0)
			//	printf("output[0] = %f output[%d] = %f\n", output[j], i, output[i]);

			output[j] += output[j + i];
		}
	}
}
//3. kahansum 수행
void CPU_kahansum(float* input, float* output, int size)
{
	int i;
	float sum = 0.0f, c = 0.0f, t, y;
	for (i = size - 1; i >= 0; i--)
	{
		y = input[i] - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}
	*output = sum;
}