#include "Reductor.hpp"
#include "my_OpenCL_util.h"

void Reductor::makeTestSet(size_t size)
{
	data_size = size;

	data_input = new float[size];
	if (data_input == nullptr)
		return;
	data_temp = new float[size];
	if (data_temp == nullptr)
		return;
	
	//항상 같은 샘플데이터가 나오게 난수표 값을 고정
	srand((unsigned int)201803);

	for (int i = 0; i < size; i++) {
		(data_input)[i] = 3.1415926f*((float)rand() / RAND_MAX);
	}
	fprintf(stdout, "---- Test Set : %d -----\n", size);

	this->getCPU()->set(data_input, data_temp, size);
	this->getGPU()->set(data_input, data_temp, size);
}
void Reductor::removeTestSet()
{
	if (data_input != nullptr)
	{
		delete data_input;
		data_input = nullptr;
	}
	if (data_temp != nullptr)
	{
		delete data_temp;
		data_temp = nullptr;
	}
	data_size = 0;
}