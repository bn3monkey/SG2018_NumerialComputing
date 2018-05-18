#ifndef __REDUCTOR_GPU__
#define __REDUCTOR_GPU__

#include "my_OpenCL_util.h"

#define COALESCED_GLOBAL_MEMORY_ACCESS 

#define INDEX_GPU_1D 0
#define INDEX_CPU_1D 1
#define INDEX_GPU_2D 2
#define INDEX_CPU_2D 3

#define NUM_DEVICE 4

enum GPU_KERNEL
{
	gpu_reduction_global,
	gpu_reduction_local,
};

typedef struct _OPENCL_C_PROG_SRC {
	size_t length;
	char *string;
} OPENCL_C_PROG_SRC;


typedef struct _preset
{
	cl_device_id device;
	cl_context context;
	cl_command_queue cmd_queue;

	cl_program program;
	cl_kernel kernel[2];

	cl_mem buffer_input;
	cl_mem buffer_temp;
	cl_mem buffer_output;

} Preset;

class Reductor_GPU
{
private:
	//OpenCL 관련 객체
	cl_platform_id platform[2];
	
	const char* source_name[2] = 
	{
		"sum_kernel_1d.cl",
		"sum_kernel_2d.cl",
	};
	OPENCL_C_PROG_SRC source[2];
	const char* kernel_name[2] =
	{
		"reduction_global",
		"reduction_local",
	};

	Preset preset[NUM_DEVICE];

	cl_event event_for_timing;

	//객체의 성공적인 생성 여부
	bool bsuccess = false;

	// host buffer
	float* input;
	float* output;
	size_t size;

	// Device 정보를 가져오는 메소드
	void getDevice();
	// 각 Device에 대하여 Context를 생성하는 메소드
	void getContext();
	// 각 Device에 대하여 Command_queue를 생성하는 메소드
	void getCommandQueue();
	//Source file을 읽어와 OpenCL program을 build하는 메소드
	void buildProgram();
	//Kernel을 가져오는 Program
	void getKernel();
	
	//Test 내부 
	//Host Buffer에 있는 내용을 Device Buffer로 가져옴.
	void transfer(int device);

	//Device Buffer에 있는 내용을 Host Buffer로 다시 가져옴.
	void receive(int device);

public :

	Reductor_GPU()
	{
		bsuccess = true;
		init();
	}

	~Reductor_GPU()
	{
		destroy();
	}

	//객체가 성공적으로 생성됐는지 알려주는 메소드
	inline bool success() { return bsuccess; }

	//OpenCL 필요변수을 초기화해주는 메소드.
	void init();
	//OpenCL 필요변수를 정리하는 메소드.
	void destroy();
	
	//host buffer를 외부에서 세팅한다.
	void set(float* input, float* output, size_t size);

	//Buffer들을 할당해주는 메소드.
	void allocBuffer();
	//Buffer들을 할당해제해주는 메소드.
	void removeBuffer();

	//1차원 데이터에서 global/local memory만을 활용하여 실험.
	void test_1d(enum GPU_KERNEL kernel_num, int trial, size_t work_group_size_gpu);
	//2차원 데이터에서 global/local memory만을 활용하여 실험.
	void test_2d(enum GPU_KERNEL kernel_num, int trial, size_t global_width, size_t work_group_width, size_t work_group_height);
	//1차원 데이터에서 CPU를 활용하여 실험.
	void test_1d_cpu(enum GPU_KERNEL kernel_num, int trial, size_t work_grou_size_cpu);
	//2차원 데이터에서 CPU를 활용하여 실험.
	void test_2d_cpu(enum GPU_KERNEL kernel_num, int trial, size_t global_width, size_t work_group_width, size_t work_group_height);
	
};

#endif