#ifndef __REDUCTOR__
#define __REDUCTOR__

#include "Reductor_GPU.hpp"
#include "Reductor_CPU.hpp"

//ERROR 메시지 객체
class ERROR_
{
public:
	const char* msg;
	ERROR_(const char* _msg) : msg(_msg) {};
};

class Reductor
{
	//각 Reductor Class에 대한 객체
	Reductor_CPU* pCPU;
	Reductor_GPU* pGPU;

	//input data 정보
	float* data_input;
	float* data_temp;
	size_t data_size;



public:
	Reductor()
	{
		pCPU = nullptr;
		pGPU = nullptr;

		pCPU = new Reductor_CPU();
		if (!pCPU->success())
			throw ERROR_("CPU Initialize ERROR!");

		pGPU = new Reductor_GPU();
		if (!pGPU->success())
			throw ERROR_("GPU Initialize ERROR!");

	}
	~Reductor()
	{
		if(pCPU != nullptr)
			delete pCPU;
		if(pGPU != nullptr)
			delete pGPU;
	}

	//inputData를 임의로 설정한다.
	void makeTestSet(size_t size);
	//inputData를 할당해제한다.
	void removeTestSet();
	//CPU 실험 객체를 얻어온다.
	inline Reductor_CPU* getCPU() { return pCPU; }
	//GPU 실험 객체를 얻어온다.
	inline Reductor_GPU* getGPU() { return pGPU; }
};

#endif