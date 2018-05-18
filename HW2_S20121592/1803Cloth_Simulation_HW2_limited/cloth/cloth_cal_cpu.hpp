#include <cstdio>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <GL/glew.h>
#include <thread>

//Fs 계산 시 4방향 계산
//#define FORTH_DIRECTION
//Fs 계산 시 8방향 계산
#define EIGHTH_DIRECTION

enum direct_part
{
	CENTER,
	LEFT,
	RIGHT,
	UP,
	DOWN,
	LEFTUP,
	LEFTDOWN,
	RIGHTUP,
	RIGHTDOWN,
};

class cloth_cal_cpu
{
	std::thread** method_thread;

	void method_Euler();
	void method_Cookbook();
	void method_SecondOrderRungeKutta();
	//void method_FourthOrderRungeKutta();

	
	

public:
	//initializeCL 대신 CPU 계산 시 들어갈 부분.
	cloth_cal_cpu(int _func_num = 0, int _thread_num = 1,
		int _NUM_ITER = 300, float _DeltaT = (1.0f / 300)*(1.0f / 60.0f), const int _NUM_PARTICLES_X = 64, const int _NUM_PARTICLES_Y = 64,
		const float _CLOTH_SIZE_X = 4.0f, const float _CLOTH_SIZE_Y = 3.0f,
		const float _PARTICLE_MASS = 0.015, const float _PARTICLE_INV_MASS = 1000 / 15,
		const float SPRING_K = 500.0f, const float _DAMPING_CONST = 0.01, const float* _Gravity = NULL);

	~cloth_cal_cpu();

	//initialize_buffer에서 CPU 계산 시 들어갈 부분에서 CL 부분만 대체
	void initialize_buffer(GLfloat*& init_position, GLfloat*& init_velocity);
	void free_buffer();

	//해당 Method 함수를 돌려서 값을 GLfloat* 형식으로 바꾸어 출력
	float* run();

};

inline int pos_index(int x, int y, int NUM_PARTICLES_X)
{
	return x + (NUM_PARTICLES_X)* y;
}
inline int pos_reindex(int x, int y, int NUM_PARTICLES_X)
{
	return (x + 1) + (NUM_PARTICLES_X + 2) * (y + 1);
}
inline int pos_direct(int i, enum direct_part d, int NUM_PARTICLES_X)
{
	int x, y;
	y = i / NUM_PARTICLES_X;
	x = i - y * NUM_PARTICLES_X;
	x += 1;
	y += 1;
	
	switch (d)
	{
	case CENTER: break;
	case LEFT: x -= 1; break;
	case RIGHT: x += 1; break;
	case UP: y -= 1; break;
	case DOWN: y +=1; break;
	case LEFTUP: x -= 1; y -= 1;  break;
	case LEFTDOWN: x -= 1; y += 1; break;
	case RIGHTUP: x += 1; y -= 1; break;
	case RIGHTDOWN: x += 1; y += 1; break;
	}

	return x + (NUM_PARTICLES_X + 2) * y;
}

#define pos_idx(x,y) pos_index(x,y, NUM_PARTICLES_X)
#define pos_reidx(x,y) pos_reindex(x, y, NUM_PARTICLES_X)

inline glm::vec4 getFs(int i, glm::vec4* pos_prev, float SPRING_K, float REST_LENGTH_HORIZ, float REST_LENGTH_VERT, float REST_LENGTH_DIAG, float NUM_PARTICLES_X, float NUM_PARTICLES_Y)
{
	int pos_idx;
	int pos_preidx;

	glm::vec4 diff;
	float distance;

	glm::vec4 Fs = glm::vec4(0);

	/*********************** Fs 계산 ****************************/
	pos_idx = pos_direct(i, CENTER, NUM_PARTICLES_X);

	
	pos_preidx = pos_direct(i, LEFT, NUM_PARTICLES_X);
	if (pos_prev[pos_preidx].w != 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = pos_prev[pos_preidx] - pos_prev[pos_idx];
		// distance = |r| 
		distance = glm::length(glm::fvec3(diff));
		// diff = r/|r|
		diff = glm::fvec4(glm::normalize(glm::fvec3(diff)), 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SPRING_K * (distance - REST_LENGTH_HORIZ)) * diff;
	}

	pos_preidx = pos_direct(i, DOWN, NUM_PARTICLES_X);
	//Fs에 결과값을 더해준다.
	if (pos_prev[pos_preidx].w != 0)
	{
		// diff = r 구하기
		diff = pos_prev[pos_preidx] - pos_prev[pos_idx];
		// distance = |r| 
		distance = glm::length(glm::fvec3(diff));
		// diff = r/|r|
		diff = glm::fvec4(glm::normalize(glm::fvec3(diff)), 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (pos_prev[pos_preidx].w * SPRING_K * (distance - REST_LENGTH_VERT)) * diff;
	}

	pos_preidx = pos_direct(i, RIGHT, NUM_PARTICLES_X);
	//Fs에 결과값을 더해준다.
	if (pos_prev[pos_preidx].w != 0)
	{
		// diff = r 구하기
		diff = pos_prev[pos_preidx] - pos_prev[pos_idx];
		// distance = |r| 
		distance = glm::length(glm::fvec3(diff));
		// diff = r/|r|
		diff = glm::fvec4(glm::normalize(glm::fvec3(diff)), 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (pos_prev[pos_preidx].w * SPRING_K * (distance - REST_LENGTH_HORIZ)) * diff;
	}

	pos_preidx = pos_direct(i, UP, NUM_PARTICLES_X);
	//Fs에 결과값을 더해준다.
	
	if (pos_prev[pos_preidx].w != 0)
	{
		// diff = r 구하기
		diff = pos_prev[pos_preidx] - pos_prev[pos_idx];
		// distance = |r| 
		distance = glm::length(glm::fvec3(diff));
		// diff = r/|r|
		diff = glm::fvec4(glm::normalize(glm::fvec3(diff)), 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (pos_prev[pos_preidx].w * SPRING_K * (distance - REST_LENGTH_VERT)) * diff;
	}

#ifdef EIGHTH_DIRECTION
	pos_preidx = pos_direct(i, RIGHTDOWN, NUM_PARTICLES_X);
	//Fs에 결과값을 더해준다.
	if (pos_prev[pos_preidx].w != 0)
	{
		// diff = r 구하기
		diff = pos_prev[pos_preidx] - pos_prev[pos_idx];
		// distance = |r| 
		distance = glm::length(glm::fvec3(diff));
		// diff = r/|r|
		diff = glm::fvec4(glm::normalize(glm::fvec3(diff)), 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (pos_prev[pos_preidx].w * SPRING_K * (distance - REST_LENGTH_DIAG)) * diff;
	}

	pos_preidx = pos_direct(i, LEFTDOWN, NUM_PARTICLES_X);
	//Fs에 결과값을 더해준다.
	if (pos_prev[pos_preidx].w != 0)
	{
		// diff = r 구하기
		diff = pos_prev[pos_preidx] - pos_prev[pos_idx];
		// distance = |r| 
		distance = glm::length(glm::fvec3(diff));
		// diff = r/|r|
		diff = glm::fvec4(glm::normalize(glm::fvec3(diff)), 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (pos_prev[pos_preidx].w * SPRING_K * (distance - REST_LENGTH_DIAG)) * diff;
	}

	pos_preidx = pos_direct(i, LEFTUP, NUM_PARTICLES_X);
	//Fs에 결과값을 더해준다.
	if (pos_prev[pos_preidx].w != 0)
	{
		// diff = r 구하기
		diff = pos_prev[pos_preidx] - pos_prev[pos_idx];
		// distance = |r| 
		distance = glm::length(glm::fvec3(diff));
		// diff = r/|r|
		diff = glm::fvec4(glm::normalize(glm::fvec3(diff)), 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (pos_prev[pos_preidx].w * SPRING_K * (distance - REST_LENGTH_DIAG)) * diff;
	}

	pos_preidx = pos_direct(i, RIGHTUP, NUM_PARTICLES_X);
	//Fs에 결과값을 더해준다.
	if (pos_prev[pos_preidx].w != 0)
	{
		// diff = r 구하기
		diff = pos_prev[pos_preidx] - pos_prev[pos_idx];
		// distance = |r| 
		distance = glm::length(glm::fvec3(diff));
		// diff = r/|r|
		diff = glm::fvec4(glm::normalize(glm::fvec3(diff)), 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (pos_prev[pos_preidx].w * SPRING_K * (distance - REST_LENGTH_DIAG)) * diff;
	}
#endif
	
	return Fs;
}
