#ifndef __REDUCTOR_CPU__
#define __REDUCTOR_CPU__

enum CPU_KERNEL
{
	cpu_simple,
	cpu_reduction,
	cpu_kahansum
};

//Kernel
//1. for 문으로 모두 더하기 수행
void CPU_simple(float* input, float* output, int size);
//2. reduction 수행 
void CPU_reduction(float* input, float* output, int size);
//3. kahansum 수행
void CPU_kahansum(float* input, float* output, int size);

class Reductor_CPU
{
private :
	bool bsuccess = false;

	//kernel 함수를 담을 배열
	void (*func_kernel[3])(float*, float*, int);
	//kernel 함수의 이름
	const char* str_kernel[3] = { "Simple", "Reduction", "KahanSum" };

	//host buffer 정보
	float* input;
	float* output;
	size_t size;

public:
	Reductor_CPU()
	{
		bsuccess = true;
		func_kernel[0] = CPU_simple;
		func_kernel[1] = CPU_reduction;
		func_kernel[2] = CPU_kahansum;
	}
	~Reductor_CPU()
	{

	}
	//객체가 성공적으로 생성됐는지 알려주는 메소드
	inline bool success() { return bsuccess; }

	//host buffer를 외부에서 세팅한다.
	void set(float* input, float* output, size_t size);
	//1차원 데이터를 CPU 환경에서 실험
	void test(enum CPU_KERNEL kernel_num, int trial);

};

#endif