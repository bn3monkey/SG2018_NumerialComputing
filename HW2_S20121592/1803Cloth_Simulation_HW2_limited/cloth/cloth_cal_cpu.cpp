#include "cloth_cal_cpu.hpp"
#include<cstring>


static int NUM_PARTICLES_X;
static int NUM_PARTICLES_Y;

static float CLOTH_SIZE_X;
static float CLOTH_SIZE_Y;

static float PARTICLE_MASS;
static float PARTICLE_INV_MASS;

static float SPRING_K;
static float REST_LENGTH_HORIZ;
static float REST_LENGTH_VERT;
static float REST_LENGTH_DIAG;

static float DAMPING_CONST;


static int buf_flag;
static glm::fvec4* position_buf[2];
//이전 position과 다음 position
static glm::fvec4* pos_prev;
static glm::fvec4* pos_next;

static glm::fvec4* velocity_buf[2];
//이전 velocity과 다음 velocity
static glm::fvec4* vel_prev;
static glm::fvec4* vel_next;

static glm::fvec4* force_buf[2];
//이전 가속도와 다음 가속도
static glm::fvec4* force_prev;
static glm::fvec4* force_next;

static glm::fvec4 Gravity;

//OpenCL buffer에 쓰기 위한 position host memory.
static float* host_position;

static int NUM_ITER;
static float DELTA_T;

//function table에서 사용할 run 함수 번호.
static int func_num;
//총 사용할 thread 개수
static int thread_num;

static void thread_method_Euler(int thread_id);
static void thread_method_Cookbook(int thread_id);
static void thread_method_SecondOrderRungeKutta(int thread_id);
static void thread_method_FourthOrderRungeKutta(int thread_id);

cloth_cal_cpu::cloth_cal_cpu(int _func_num, int _thread_num,
	int _NUM_ITER, float _DeltaT, const int _NUM_PARTICLES_X , const int _NUM_PARTICLES_Y,
	const float _CLOTH_SIZE_X, const float _CLOTH_SIZE_Y,
	const float _PARTICLE_MASS, const float _PARTICLE_INV_MASS,
	const float _SPRING_K, const float _DAMPING_CONST, const float* _Gravity)
{
	func_num = _func_num;
	thread_num = _thread_num;
	NUM_ITER = _NUM_ITER;
	DELTA_T = _DeltaT;

	NUM_PARTICLES_X = _NUM_PARTICLES_X;
	NUM_PARTICLES_Y = _NUM_PARTICLES_Y;
	CLOTH_SIZE_X = _CLOTH_SIZE_X;
	CLOTH_SIZE_Y = _CLOTH_SIZE_Y;
	PARTICLE_MASS = _PARTICLE_MASS;
	PARTICLE_INV_MASS = _PARTICLE_INV_MASS;
	SPRING_K = _SPRING_K;
	DAMPING_CONST = _DAMPING_CONST;

	if (_Gravity == NULL)
	{
		Gravity.y = -9.80665f;
		Gravity.x = Gravity.z = Gravity.w = 0;
	}
	else
	{
		Gravity.x = _Gravity[0];
		Gravity.y = _Gravity[1];
		Gravity.z = _Gravity[2];
		Gravity.w = _Gravity[3];
	}
	REST_LENGTH_HORIZ = CLOTH_SIZE_X / (NUM_PARTICLES_X - 1);
	REST_LENGTH_VERT = CLOTH_SIZE_Y / (NUM_PARTICLES_Y - 1);
	REST_LENGTH_DIAG = sqrtf(REST_LENGTH_HORIZ*REST_LENGTH_HORIZ + REST_LENGTH_VERT * REST_LENGTH_VERT);

	pos_prev = NULL;
	pos_next = NULL;
	vel_prev = NULL;
	vel_next = NULL;

	method_thread = new std::thread*[thread_num];
	if (method_thread == NULL)
	{
		fprintf(stderr, "thread pointer allocation fail!\n");
		exit(-1);
	}
}

cloth_cal_cpu::~cloth_cal_cpu()
{
	free_buffer();
	if (method_thread != NULL)
		delete method_thread;
}

void cloth_cal_cpu::initialize_buffer(GLfloat*& init_position, GLfloat*& init_velocity)
{
	// Initial transform
	glm::mat4 transf = glm::translate(glm::mat4(1.0), glm::vec3(0, CLOTH_SIZE_Y, 0));
	transf = glm::rotate(transf, glm::radians(-80.0f), glm::vec3(1, 0, 0));
	transf = glm::translate(transf, glm::vec3(0, -CLOTH_SIZE_Y, 0));

	// Initial positions of the particles
	buf_flag = 0;
	int buffer_size = NUM_PARTICLES_X * NUM_PARTICLES_Y;
	pos_prev = position_buf[0] = new glm::vec4[buffer_size];
	if (pos_prev == NULL)
	{
		free_buffer();
		exit(-1);
	}
	pos_next = position_buf[1] = new glm::vec4[buffer_size];
	if (pos_next == NULL)
	{
		free_buffer();
		exit(-1);
	}

		vel_prev = velocity_buf[0] = new glm::vec4[buffer_size];
	if (vel_prev == NULL)
	{
		free_buffer();
		exit(-1);
	}
	vel_next = velocity_buf[1] = new glm::vec4[buffer_size];
	if (vel_next == NULL)
	{
		free_buffer();
		exit(-1);
	}
	for (int i = 0; i < buffer_size; i++)
		vel_prev[i] = glm::vec4(0);
	
	force_prev = force_buf[0] = new glm::vec4[buffer_size];
	if (force_prev == NULL)
	{
		free_buffer();
		exit(-1);
	}
	force_next = force_buf[1] = new glm::vec4[buffer_size];
	if (force_next == NULL)
	{
		free_buffer();
		exit(-1);
	}
	for (int i = 0; i < buffer_size; i++)
		force_prev[i] = glm::vec4(0);


	host_position = new float[4 * buffer_size];

	float dx = CLOTH_SIZE_X / (NUM_PARTICLES_X - 1);
	float dy = CLOTH_SIZE_Y / (NUM_PARTICLES_Y - 1);
	float ds = 1.0f / (NUM_PARTICLES_X - 1);
	float dt = 1.0f / (NUM_PARTICLES_Y - 1);

	glm::vec4 p(0.0f, 0.0f, 0.0f, 1.0f);
	int idx;
	for (int i = 0; i < NUM_PARTICLES_Y; i++) {
		for (int j = 0; j < NUM_PARTICLES_X; j++) {
			p.x = dx * j;
			p.y = dy * i;
			p.z = 0.0f;
			p = transf * p;

			idx = pos_idx(j, i);
			pos_prev[idx] = p;

			init_position[idx] = p.x;
			init_position[idx] = p.y;
			init_position[idx] = p.z;
			init_position[idx] = p.w;
		}
	}
}
void cloth_cal_cpu::free_buffer()
{
	if (pos_prev != NULL)
		delete pos_prev;
	if (pos_next != NULL)
		delete pos_next;
	if (vel_prev != NULL)
		delete vel_prev;
	if (vel_next != NULL)
		delete vel_next;
	if (force_prev != NULL)
		delete force_prev;
	if (force_next != NULL)
		delete force_next;
};

float* cloth_cal_cpu::run()
{
	switch (func_num)
	{
		case 0: method_Euler(); break;
		case 1: method_Cookbook(); break;
		case 2: method_SecondOrderRungeKutta(); break;
		//case 3: method_FourthOrderRungeKutta(); break;
	}

	int buffer_size = NUM_PARTICLES_X * NUM_PARTICLES_Y;
	for(int arr_idx = 0 ; arr_idx < buffer_size ; arr_idx++)
	{
		host_position[4 * arr_idx] = pos_prev[arr_idx].x;
		host_position[4 * arr_idx + 1] = pos_prev[arr_idx].y;
		host_position[4 * arr_idx + 2] = pos_prev[arr_idx].z;
		host_position[4 * arr_idx + 3] = pos_prev[arr_idx].w;
	}

	return host_position;
}


void cloth_cal_cpu::method_Euler()
{
	//5개의 pin의 위치 index
	int first = pos_idx(0, NUM_PARTICLES_Y - 1);
	int second = pos_idx(NUM_PARTICLES_X >> 2, NUM_PARTICLES_Y - 1);
	int third = pos_idx(NUM_PARTICLES_X >> 1, NUM_PARTICLES_Y - 1);
	int forth = second + third - first;
	int fifth = pos_idx(NUM_PARTICLES_X -1, NUM_PARTICLES_Y - 1);

	for (int iter = 0; iter < NUM_ITER; iter++)
	{
		pos_prev = position_buf[buf_flag];
		pos_next = position_buf[1 - buf_flag];

		vel_prev = velocity_buf[buf_flag];
		vel_next = velocity_buf[1 - buf_flag];

		force_prev = force_buf[buf_flag];
		force_next = force_buf[1 - buf_flag];

		for (int i = 0; i < thread_num; i++)
			method_thread[i] = new std::thread(thread_method_Euler, i);

		for (int i = 0; i < thread_num; i++)
			method_thread[i]->join();

		//pin으로 고정한 것은 지속적으로 롤백.
		pos_next[first] = pos_prev[first];
		pos_next[second] = pos_prev[second];
		pos_next[third] = pos_prev[third];
		pos_next[forth] = pos_prev[forth];
		pos_next[fifth] = pos_prev[fifth];

		buf_flag = 1 - buf_flag;
	}
}


void thread_method_Euler(int thread_id)
{
	int buffer_totalsize = NUM_PARTICLES_X * NUM_PARTICLES_Y;
	int buffer_size = buffer_totalsize / thread_num;
	int start = thread_id * buffer_size;
	int end = thread_id * buffer_size + buffer_size;

	glm::fvec4 Fs;
	glm::fvec4 Fg;
	glm::fvec4 Fd;
	glm::fvec4 a;
	
	for (int i = start; i < end; i++)
	{
		/*********************** Fs 계산 ****************************/
		Fs = getFs(i, pos_prev, SPRING_K, REST_LENGTH_HORIZ, REST_LENGTH_VERT, REST_LENGTH_DIAG, NUM_PARTICLES_X, NUM_PARTICLES_Y);


		/*********************** Fg 계산 ****************************/
		Fg = PARTICLE_MASS * Gravity;

		/*********************** Fd 계산 ****************************/
		Fd = -DAMPING_CONST * vel_prev[i];

		/*********************** 다음 힘 계산 ****************************/
		force_next[i] = Fs + Fg + Fd;
		
		/*********************** 가속도 계산 ****************************/
		a = force_prev[i] * PARTICLE_INV_MASS;

		/*********************** First Order 계산 ****************************/
		vel_next[i] = vel_prev[i] + a * DELTA_T;
		pos_next[i] = pos_prev[i] + vel_prev[i] * DELTA_T;

	}
}

void cloth_cal_cpu::method_Cookbook()
{
	//5개의 pin의 위치 index
	int first = pos_idx(0, NUM_PARTICLES_Y - 1);
	int second = pos_idx(NUM_PARTICLES_X >> 2, NUM_PARTICLES_Y - 1);
	int third = pos_idx(NUM_PARTICLES_X >> 1, NUM_PARTICLES_Y - 1);
	int forth = second + third - first;
	int fifth = pos_idx(NUM_PARTICLES_X - 1, NUM_PARTICLES_Y - 1);

	for (int iter = 0; iter < NUM_ITER; iter++)
	{
		pos_prev = position_buf[buf_flag];
		pos_next = position_buf[1 - buf_flag];

		vel_prev = velocity_buf[buf_flag];
		vel_next = velocity_buf[1 - buf_flag];

		force_prev = force_buf[buf_flag];
		force_next = force_buf[1 - buf_flag];

		for (int i = 0; i < thread_num; i++)
			method_thread[i] = new std::thread(thread_method_Cookbook, i);

		for (int i = 0; i < thread_num; i++)
			method_thread[i]->join();

		//pin으로 고정한 것은 지속적으로 롤백.
		pos_next[first] = pos_prev[first];
		pos_next[second] = pos_prev[second];
		pos_next[third] = pos_prev[third];
		pos_next[forth] = pos_prev[forth];
		pos_next[fifth] = pos_prev[fifth];

		buf_flag = 1 - buf_flag;
	}

}


void thread_method_Cookbook(int thread_id)
{
	int buffer_totalsize = NUM_PARTICLES_X * NUM_PARTICLES_Y;
	int buffer_size = buffer_totalsize / thread_num;
	int start = thread_id * buffer_size;
	int end = thread_id * buffer_size + buffer_size;

	glm::fvec4 Fs;
	glm::fvec4 Fg;
	glm::fvec4 Fd;
	glm::fvec4 a;

	for (int i = start; i < end; i++)
	{
		/*********************** Fs 계산 ****************************/
		Fs = getFs(i, pos_prev, SPRING_K, REST_LENGTH_HORIZ, REST_LENGTH_VERT, REST_LENGTH_DIAG, NUM_PARTICLES_X, NUM_PARTICLES_Y);
		/*********************** Fg 계산 ****************************/
		Fg = PARTICLE_MASS * Gravity;

		/*********************** Fd 계산 ****************************/
		Fd = -DAMPING_CONST * vel_prev[i];

		/*********************** 다음 힘 계산 ****************************/
		force_next[i] = Fs + Fg + Fd;

		/*********************** 가속도 계산 ****************************/
		a = force_prev[i] * PARTICLE_INV_MASS;

		/*********************** First Order 계산 ****************************/
		vel_next[i] = vel_prev[i] + a * DELTA_T;
		pos_next[i] = pos_prev[i] + (vel_prev[i] + 0.5f * a * DELTA_T) * DELTA_T;

	}
}

void cloth_cal_cpu::method_SecondOrderRungeKutta()
{
	//5개의 pin의 위치 index
	int first = pos_idx(0, NUM_PARTICLES_Y - 1);
	int second = pos_idx(NUM_PARTICLES_X >> 2, NUM_PARTICLES_Y - 1);
	int third = pos_idx(NUM_PARTICLES_X >> 1, NUM_PARTICLES_Y - 1);
	int forth = second + third - first;
	int fifth = pos_idx(NUM_PARTICLES_X - 1, NUM_PARTICLES_Y - 1);

	for (int iter = 0; iter < NUM_ITER; iter++)
	{
		pos_prev = position_buf[buf_flag];
		pos_next = position_buf[1 - buf_flag];

		vel_prev = velocity_buf[buf_flag];
		vel_next = velocity_buf[1 - buf_flag];

		force_prev = force_buf[buf_flag];
		force_next = force_buf[1 - buf_flag];

		for (int i = 0; i < thread_num; i++)
			method_thread[i] = new std::thread(thread_method_SecondOrderRungeKutta, i);

		for (int i = 0; i < thread_num; i++)
			method_thread[i]->join();

		//pin으로 고정한 것은 지속적으로 롤백.
		pos_next[first] = pos_prev[first];
		pos_next[second] = pos_prev[second];
		pos_next[third] = pos_prev[third];
		pos_next[forth] = pos_prev[forth];
		pos_next[fifth] = pos_prev[fifth];

		buf_flag = 1 - buf_flag;
	}

}


void thread_method_SecondOrderRungeKutta(int thread_id)
{
	int buffer_totalsize = NUM_PARTICLES_X * NUM_PARTICLES_Y;
	int buffer_size = buffer_totalsize / thread_num;
	int start = thread_id * buffer_size;
	int end = thread_id * buffer_size + buffer_size;

	glm::fvec4 Fs;
	glm::fvec4 Fg;
	glm::fvec4 Fd;
	glm::fvec4 nexta;
	glm::fvec4 a;

	for (int i = start; i < end; i++)
	{
		/*********************** Fs 계산 ****************************/
		Fs = getFs(i, pos_prev, SPRING_K, REST_LENGTH_HORIZ, REST_LENGTH_VERT, REST_LENGTH_DIAG, NUM_PARTICLES_X, NUM_PARTICLES_Y);
		/*********************** Fg 계산 ****************************/
		Fg = PARTICLE_MASS * Gravity;

		/*********************** Fd 계산 ****************************/
		Fd = -DAMPING_CONST * vel_prev[i];

		/*********************** Fsum 계산 ****************************/
		force_next[i] = Fs + Fg + Fd;

		/*********************** 가속도 계산 ****************************/
		a = force_prev[i] * PARTICLE_INV_MASS;
		nexta = force_next[i] * PARTICLE_INV_MASS;

		/*********************** First Order 계산 ****************************/
		vel_next[i] = vel_prev[i] + 0.5f * (a + nexta) * DELTA_T;
		pos_next[i] = pos_prev[i] + (vel_prev[i] + 0.5f * a * DELTA_T) * DELTA_T;

	}
}