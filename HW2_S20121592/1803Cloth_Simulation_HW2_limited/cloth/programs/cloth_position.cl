enum direct_part
{
	LEFTUP,
	UP,
	RIGHTUP,
	LEFT,
	CENTER,
	RIGHT,
	LEFTDOWN,
	DOWN,
	RIGHTDOWN,
};

inline int direct(int x, int y, enum direct_part d, int NUM_PARTICLES_X, int NUM_PARTICLES_Y)
{
	switch (d)
	{
	case CENTER: break;
	case LEFT: x -= 1; break;
	case RIGHT: x += 1; break;
	case UP: y -= 1; break;
	case DOWN: y += 1; break;
	case LEFTUP: x -= 1; y -= 1; break;
	case LEFTDOWN:x -= 1; y += 1; break;
	case RIGHTUP: x += 1; y -= 1; break;
	case RIGHTDOWN: x += 1; y += 1; break;
	}

	if (x < 0 || x >= NUM_PARTICLES_X || y < 0 || y >= NUM_PARTICLES_Y)
		return -1;
	else
		return y * NUM_PARTICLES_X + x;
}

inline float4 getFs(int x, int y, __global float4* pos_in, float SpringK, float RestLengthHoriz, float RestLengthVert, float RestLengthDiag, float NUM_PARTICLES_X, float NUM_PARTICLES_Y)
{
	int idx;
	float4 diff;
	float distance;
	//계산하기 전 Fs는 Clear
	float4 Fs = (float4)(0);

	int i = x + NUM_PARTICLES_X * y;
	idx = direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = pos_in[idx] - pos_in[i];
		// distance = |r| 
		distance = fast_length(diff.xyz);

		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);

		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthHoriz)) * diff;
	}
	idx = direct(x, y, DOWN, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = pos_in[idx] - pos_in[i];
		// distance = |r| 
		distance = fast_length(diff.xyz);

		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);

		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthVert)) * diff;
	}

	idx = direct(x, y, RIGHT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = pos_in[idx] - pos_in[i];
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthHoriz)) * diff;
	}
	idx = direct(x, y, UP, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기s
		diff = pos_in[idx] - pos_in[i];
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthVert)) * diff;
	}

	idx = direct(x, y, RIGHTDOWN, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = pos_in[idx] - pos_in[i];
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthDiag)) * diff;
	}
	idx = direct(x, y, LEFTDOWN, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = pos_in[idx] - pos_in[i];
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthDiag)) * diff;
	}
	idx = direct(x, y, LEFTUP, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = pos_in[idx] - pos_in[i];
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthDiag)) * diff;
	}
	idx = direct(x, y, RIGHTUP, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = pos_in[idx] - pos_in[i];
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthDiag)) * diff;
	}

	return Fs;
}

#define new_pos(i) (pos_in[i] + vel_in[i] * DELTA_T)
inline float4 getnewFs(int x, int y, __global float4* pos_in, __global float4* vel_in, float DELTA_T, float SpringK, float RestLengthHoriz, float RestLengthVert, float RestLengthDiag, float NUM_PARTICLES_X, float NUM_PARTICLES_Y)
{
	int idx;
	float4 diff;
	float distance;
	//계산하기 전 Fs는 Clear
	float4 Fs = (float4)(0);

	int i = x + NUM_PARTICLES_X * y;
	idx = direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = new_pos(idx) - new_pos(i);
		// distance = |r| 
		distance = fast_length(diff.xyz);

		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);

		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthHoriz)) * diff;
	}
	idx = direct(x, y, DOWN, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = new_pos(idx) - new_pos(i);
		// distance = |r| 
		distance = fast_length(diff.xyz);

		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);

		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthVert)) * diff;
	}

	idx = direct(x, y, RIGHT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = new_pos(idx) - new_pos(i);
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthHoriz)) * diff;
	}
	idx = direct(x, y, UP, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기s
		diff = new_pos(idx) - new_pos(i);
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthVert)) * diff;
	}

	idx = direct(x, y, RIGHTDOWN, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = new_pos(idx) - new_pos(i);
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthDiag)) * diff;
	}
	idx = direct(x, y, LEFTDOWN, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = new_pos(idx) - new_pos(i);
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthDiag)) * diff;
	}
	idx = direct(x, y, LEFTUP, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = new_pos(idx) - new_pos(i);
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthDiag)) * diff;
	}
	idx = direct(x, y, RIGHTUP, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = new_pos(idx) - new_pos(i);
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthDiag)) * diff;
	}

	return Fs;
}

__kernel
void cloth_position_global_euler(
    __global float4* pos_in, __global float4* pos_out,
    __global float4* vel_in, __global float4* vel_out,
	__global float4* force_in, __global float4* force_out,
    __local float4* local_data,
    float3 Gravity,
    float ParticleMass,
    float ParticleInvMass,
    float SpringK,
    float RestLengthHoriz,
    float RestLengthVert,
    float RestLengthDiag,
    float DeltaT,
    float DampingConst) {

	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
    int idx = x + width * y;
	float4 Fs, Fg, Fd, a;

	/*********************** Fs 계산 ****************************/
	Fs = getFs(x, y, pos_in, SpringK, RestLengthHoriz, RestLengthVert, RestLengthDiag, width, height);

	/*********************** Fg 계산 ****************************/
	Fg = ParticleMass * (float4)(Gravity, 0.0f);

	/*********************** Fd 계산 ****************************/
	Fd = -DampingConst * vel_in[idx];

	/*********************** 다음 힘 계산 ****************************/
	Fs = Fs + Fg + Fd;

	/*********************** 가속도 계산 ****************************/
	a = Fs * ParticleInvMass;

	/*********************** First Order 계산 ****************************/
	vel_out[idx] = vel_in[idx] + a * DeltaT;
	pos_out[idx] = pos_in[idx] + vel_out[idx] * DeltaT;


	if (y == height -1)
	{
		int span = (width >> 2);
		int span2 = span + span;
		int span3 = span + span2;
		int span4 = width - 1;
		
		if (x == 0 || x == span || x == span2 || x == span3 || x == span4)
		{
			vel_out[idx] = vel_in[idx];
			pos_out[idx] = pos_in[idx];
		}
	}

}

__kernel
void cloth_position_global_cookbook(
	__global float4* pos_in, __global float4* pos_out,
	__global float4* vel_in, __global float4* vel_out,
	__global float4* force_in, __global float4* force_out,
	__local float4* local_data,
	float3 Gravity,
	float ParticleMass,
	float ParticleInvMass,
	float SpringK,
	float RestLengthHoriz,
	float RestLengthVert,
	float RestLengthDiag,
	float DeltaT,
	float DampingConst) {

	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int idx = x + width * y;
	float4 Fs, Fg, Fd, a;

	/*********************** Fs 계산 ****************************/
	Fs = getFs(x, y, pos_in, SpringK, RestLengthHoriz, RestLengthVert, RestLengthDiag, width, height);

	/*********************** Fg 계산 ****************************/
	Fg = ParticleMass * (float4)(Gravity, 0.0f);

	/*********************** Fd 계산 ****************************/
	Fd = -DampingConst * vel_in[idx];

	/*********************** 다음 힘 계산 ****************************/
	Fs = Fs + Fg + Fd;

	/*********************** 가속도 계산 ****************************/
	a = Fs * ParticleInvMass;

	/*********************** First Order 계산 ****************************/
	vel_out[idx] = vel_in[idx] + a * DeltaT;
	pos_out[idx] = pos_in[idx] + DeltaT * (vel_out[idx] + 0.5f * DeltaT * a);


	if (y == height - 1)
	{
		int span = (width >> 2);
		int span2 = span + span;
		int span3 = span + span2;
		int span4 = width - 1;

		if (x == 0 || x == span || x == span2 || x == span3 || x == span4)
		{
			vel_out[idx] = vel_in[idx];
			pos_out[idx] = pos_in[idx];
		}
	}

}

__kernel
void cloth_position_global_modified(
	__global float4* pos_in, __global float4* pos_out,
	__global float4* vel_in, __global float4* vel_out,
	__global float4* force_in, __global float4* force_out,
	__local float4* local_data,
	float3 Gravity,
	float ParticleMass,
	float ParticleInvMass,
	float SpringK,
	float RestLengthHoriz,
	float RestLengthVert,
	float RestLengthDiag,
	float DeltaT,
	float DampingConst) {

	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int idx = x + width * y;
	float4 Fs, Fg, Fd, a, nexta;

	/*********************** Fs 계산 ****************************/
	Fs = getFs(x, y, pos_in, SpringK, RestLengthHoriz, RestLengthVert, RestLengthDiag, width, height);

	/*********************** Fg 계산 ****************************/
	Fg = ParticleMass * (float4)(Gravity, 0.0f);

	/*********************** Fd 계산 ****************************/
	Fd = -DampingConst * vel_in[idx];

	/*********************** 다음 힘 계산 ****************************/
	force_in[idx] = Fs + Fg + Fd;
	/*********************** 가속도 계산 ****************************/
	a = force_in[idx] * ParticleInvMass;

	/*********************** Fs 계산 ****************************/
	Fs = getnewFs(x, y, pos_in, vel_in, DeltaT, SpringK, RestLengthHoriz, RestLengthVert, RestLengthDiag, width, height);

	/*********************** Fg 계산 ****************************/
	Fg = ParticleMass * (float4)(Gravity, 0.0f);

	/*********************** Fd 계산 ****************************/
	Fd = -DampingConst * (vel_in[idx] + a * DeltaT);

	/*********************** 다음 힘 계산 ****************************/
	force_out[idx] = Fs + Fg + Fd;
	nexta = force_out[idx] * ParticleInvMass;

	/*********************** First Order 계산 ****************************/
	vel_out[idx] = vel_in[idx] + 0.5f * (a + nexta)* DeltaT;
	pos_out[idx] = pos_in[idx] + 0.5f * (vel_in[idx] + vel_out[idx]) * DeltaT;


	if (y == height - 1)
	{
		int span = (width >> 2);
		int span2 = span + span;
		int span3 = span + span2;
		int span4 = width - 1;

		if (x == 0 || x == span || x == span2 || x == span3 || x == span4)
		{
			vel_out[idx] = vel_in[idx];
			pos_out[idx] = pos_in[idx];
			force_out[idx] = force_in[idx];
		}
	}

}


inline void move_global_to_local(int idx, __global float4* pos_in, __local float4* local_data)
{
	// Copy into local memory
	uint local_width = get_local_size(0) + 2;

	local_data[(get_local_id(1) + 1) * local_width + (get_local_id(0) + 1)] = pos_in[idx];

	// Bottom edge
	if (get_local_id(1) == 0)
	{
		if (get_global_id(1) > 0)
		{
			local_data[get_local_id(0) + 1] = pos_in[idx - get_global_size(0)];

			// Lower left corner
			if (get_local_id(0) == 0)
			{
				if (get_global_id(0) > 0)
				{
					local_data[0] = pos_in[idx - get_global_size(0) - 1];
				}
				else
				{
					local_data[0] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
				}
			}

			// Lower right corner
			if (get_local_id(0) == get_local_size(0) - 1)
			{
				if (get_global_id(0) < get_global_size(0) - 1)
				{
					local_data[get_local_size(0) + 1] = pos_in[idx - get_global_size(0) + 1];
				}
				else
				{
					local_data[get_local_size(0) + 1] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
				}
			}
		}
		else
		{
			local_data[get_local_id(0) + 1] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}

	// Top edge
	if (get_local_id(1) == get_local_size(1) - 1)
	{
		if (get_global_id(1) < get_global_size(1) - 1)
		{
			local_data[(get_local_size(1) + 1) * local_width + (get_local_id(0) + 1)] = pos_in[idx + get_global_size(0)];

			// Upper left corner
			if (get_local_id(0) == 0)
			{
				if (get_global_id(0) > 0)
				{
					local_data[(get_local_size(1) + 1) * local_width] = pos_in[idx + get_global_size(0) - 1];
				}
				else
				{
					local_data[(get_local_size(1) + 1) * local_width] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
				}
			}

			//Lower right corner
			if (get_local_id(0) == get_local_size(0) - 1)
			{
				if (get_global_id(0) < get_global_size(0) - 1)
				{
					local_data[(get_local_size(1) + 1) * local_width + (get_local_size(0) + 1)] = pos_in[idx + get_global_size(0) + 1];
				}
				else
				{
					local_data[(get_local_size(1) + 1) * local_width + (get_local_size(0) + 1)] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
				}
			}
		}
		else
		{
			local_data[(get_local_size(1) + 1) * local_width + (get_local_id(0) + 1)] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}

	// Left edge
	if (get_local_id(0) == 0)
	{
		if (get_global_id(0) > 0)
		{
			local_data[(get_local_id(1) + 1) * local_width] = pos_in[idx - 1];
		}
		else
		{
			local_data[(get_local_id(1) + 1) * local_width] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}

	// Right edge
	if (get_local_id(0) == get_local_size(0) - 1)
	{
		if (get_global_id(0) < get_global_size(0) - 1)
		{
			local_data[(get_local_id(1) + 1) * local_width + (get_local_size(0) + 1)] = pos_in[idx + 1];
		}
		else
		{
			local_data[(get_local_id(1) + 1) * local_width + (get_local_size(0) + 1)] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
}

inline int local_direct(int x, int y, enum direct_part d, int NUM_PARTICLES_X, int NUM_PARTICLES_Y)
{

	int local_x = get_local_id(0) + 1;
	int local_y = get_local_id(1) + 1;

	switch (d)
	{
	case CENTER: break;
	case LEFT: x -= 1; local_x -= 1; break;
	case RIGHT: x += 1; local_x += 1; break;
	case UP: y -= 1; local_y -= 1; break;
	case DOWN: y += 1; local_y += 1; break;
	case LEFTUP: x -= 1; y -= 1; local_x -= 1; local_y -= 1; break;
	case LEFTDOWN:x -= 1; y += 1; local_x -= 1; local_y += 1; break;
	case RIGHTUP: x += 1; y -= 1; local_x += 1; local_y -= 1; break;
	case RIGHTDOWN: x += 1; y += 1; local_x += 1; local_y += 1; break;
	}

	if (x < 0 || x >= NUM_PARTICLES_X || y < 0 || y >= NUM_PARTICLES_Y)
		return -1;
	
	return local_y * (get_local_size(0) + 2) + local_x;
}

inline int local_refresh(int idx, __local float4* local_data, __global float4* pos_in, __global float4* vel_in, float DeltaT)
{
	uint local_width = get_local_size(0) + 2;

	local_data[(get_local_id(1) + 1) * local_width + (get_local_id(0) + 1)] = pos_in[idx] + vel_in[idx] * DeltaT;

	// Bottom edge
	if (get_local_id(1) == 0)
	{
		if (get_global_id(1) > 0)
		{
			local_data[get_local_id(0) + 1] = pos_in[idx - get_global_size(0)] + vel_in[idx - get_global_size(0)] * DeltaT;

			// Lower left corner
			if (get_local_id(0) == 0)
			{
				if (get_global_id(0) > 0)
				{
					local_data[0] = pos_in[idx - get_global_size(0) - 1] + vel_in[idx - get_global_size(0) - 1] * DeltaT;
				}
				else
				{
					local_data[0] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
				}
			}

			// Lower right corner
			if (get_local_id(0) == get_local_size(0) - 1)
			{
				if (get_global_id(0) < get_global_size(0) - 1)
				{
					local_data[get_local_size(0) + 1] = pos_in[idx - get_global_size(0) + 1] + vel_in[idx - get_global_size(0) + 1] * DeltaT;
				}
				else
				{
					local_data[get_local_size(0) + 1] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
				}
			}
		}
		else
		{
			local_data[get_local_id(0) + 1] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}

	// Top edge
	if (get_local_id(1) == get_local_size(1) - 1)
	{
		if (get_global_id(1) < get_global_size(1) - 1)
		{
			local_data[(get_local_size(1) + 1) * local_width + (get_local_id(0) + 1)] = pos_in[idx + get_global_size(0)] + vel_in[idx + get_global_size(0)] * DeltaT;

			// Upper left corner
			if (get_local_id(0) == 0)
			{
				if (get_global_id(0) > 0)
				{
					local_data[(get_local_size(1) + 1) * local_width] = pos_in[idx + get_global_size(0) - 1] + vel_in[idx + get_global_size(0) - 1] * DeltaT;
				}
				else
				{
					local_data[(get_local_size(1) + 1) * local_width] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
				}
			}

			//Lower right corner
			if (get_local_id(0) == get_local_size(0) - 1)
			{
				if (get_global_id(0) < get_global_size(0) - 1)
				{
					local_data[(get_local_size(1) + 1) * local_width + (get_local_size(0) + 1)] = pos_in[idx + get_global_size(0) + 1] + vel_in[idx + get_global_size(0) + 1] * DeltaT;
				}
				else
				{
					local_data[(get_local_size(1) + 1) * local_width + (get_local_size(0) + 1)] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
				}
			}
		}
		else
		{
			local_data[(get_local_size(1) + 1) * local_width + (get_local_id(0) + 1)] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}

	// Left edge
	if (get_local_id(0) == 0)
	{
		if (get_global_id(0) > 0)
		{
			local_data[(get_local_id(1) + 1) * local_width] = pos_in[idx - 1] + vel_in[idx - 1] * DeltaT;
		}
		else
		{
			local_data[(get_local_id(1) + 1) * local_width] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}

	// Right edge
	if (get_local_id(0) == get_local_size(0) - 1)
	{
		if (get_global_id(0) < get_global_size(0) - 1)
		{
			local_data[(get_local_id(1) + 1) * local_width + (get_local_size(0) + 1)] = pos_in[idx + 1] + vel_in[idx + 1] * DeltaT;
		}
		else
		{
			local_data[(get_local_id(1) + 1) * local_width + (get_local_size(0) + 1)] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
}

inline float4 getlocalFs(int x, int y, __local float4* pos_in, float SpringK, float RestLengthHoriz, float RestLengthVert, float RestLengthDiag, float NUM_PARTICLES_X, float NUM_PARTICLES_Y)
{
	int idx;
	float4 diff;
	float distance;
	//계산하기 전 Fs는 Clear
	float4 Fs = (float4)(0);

	int i = local_direct(x, y, CENTER, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	idx = local_direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = pos_in[idx] - pos_in[i];
		// distance = |r| 
		distance = fast_length(diff.xyz);

		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);

		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthHoriz)) * diff;
	}
	idx = local_direct(x, y, DOWN, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = pos_in[idx] - pos_in[i];
		// distance = |r| 
		distance = fast_length(diff.xyz);

		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);

		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthVert)) * diff;
	}

	idx = local_direct(x, y, RIGHT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = pos_in[idx] - pos_in[i];
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthHoriz)) * diff;
	}
	idx = local_direct(x, y, UP, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기s
		diff = pos_in[idx] - pos_in[i];
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthVert)) * diff;
	}

	idx = local_direct(x, y, RIGHTDOWN, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = pos_in[idx] - pos_in[i];
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthDiag)) * diff;
	}
	idx = local_direct(x, y, LEFTDOWN, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = pos_in[idx] - pos_in[i];
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthDiag)) * diff;
	}
	idx = local_direct(x, y, LEFTUP, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = pos_in[idx] - pos_in[i];
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthDiag)) * diff;
	}
	idx = local_direct(x, y, RIGHTUP, NUM_PARTICLES_X, NUM_PARTICLES_Y);
	if (idx >= 0)
	{
		//Fs에 결과값을 더해준다.
		// diff = r 구하기
		diff = pos_in[idx] - pos_in[i];
		// distance = |r| 
		distance = fast_length(diff.xyz);
		// diff = r/|r|
		diff = (float4)(diff.xyz / distance, 0.0f);
		// F = K *(|r| - R)*(r/|r|)
		Fs += (SpringK * (distance - RestLengthDiag)) * diff;
	}

	return Fs;
}

__kernel
void cloth_position_local_euler(
	__global float4* pos_in, __global float4* pos_out,
	__global float4* vel_in, __global float4* vel_out,
	__global float4* force_in, __global float4* force_out,
	__local float4* local_data,
	float3 Gravity,
	float ParticleMass,
	float ParticleInvMass,
	float SpringK,
	float RestLengthHoriz,
	float RestLengthVert,
	float RestLengthDiag,
	float DeltaT,
	float DampingConst) {

	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int idx = x + width * y;
	float4 Fs, Fg, Fd, a;

	move_global_to_local(idx, pos_in, local_data);

	/*********************** Fs 계산 ****************************/
	Fs = getlocalFs(x, y, local_data, SpringK, RestLengthHoriz, RestLengthVert, RestLengthDiag, width, height);

	/*********************** Fg 계산 ****************************/
	Fg = ParticleMass * (float4)(Gravity, 0.0f);

	/*********************** Fd 계산 ****************************/
	Fd = -DampingConst * vel_in[idx];

	/*********************** 다음 힘 계산 ****************************/
	Fs = Fs + Fg + Fd;

	/*********************** 가속도 계산 ****************************/
	a = Fs * ParticleInvMass;

	/*********************** First Order 계산 ****************************/
	vel_out[idx] = vel_in[idx] + a * DeltaT;
	pos_out[idx] = pos_in[idx] + vel_out[idx] * DeltaT;


	if (y == height - 1)
	{
		int span = (width >> 2);
		int span2 = span + span;
		int span3 = span + span2;
		int span4 = width - 1;

		if (x == 0 || x == span || x == span2 || x == span3 || x == span4)
		{
			vel_out[idx] = vel_in[idx];
			pos_out[idx] = pos_in[idx];
		}
	}

}

__kernel
void cloth_position_local_cookbook(
	__global float4* pos_in, __global float4* pos_out,
	__global float4* vel_in, __global float4* vel_out,
	__global float4* force_in, __global float4* force_out,
	__local float4* local_data,
	float3 Gravity,
	float ParticleMass,
	float ParticleInvMass,
	float SpringK,
	float RestLengthHoriz,
	float RestLengthVert,
	float RestLengthDiag,
	float DeltaT,
	float DampingConst) {

	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int idx = x + width * y;
	float4 Fs, Fg, Fd, a;

	move_global_to_local(idx, pos_in, local_data);

	/*********************** Fs 계산 ****************************/
	Fs = getlocalFs(x, y, local_data, SpringK, RestLengthHoriz, RestLengthVert, RestLengthDiag, width, height);

	/*********************** Fg 계산 ****************************/
	Fg = ParticleMass * (float4)(Gravity, 0.0f);

	/*********************** Fd 계산 ****************************/
	Fd = -DampingConst * vel_in[idx];

	/*********************** 다음 힘 계산 ****************************/
	Fs = Fs + Fg + Fd;

	/*********************** 가속도 계산 ****************************/
	a = Fs * ParticleInvMass;

	/*********************** First Order 계산 ****************************/
	vel_out[idx] = vel_in[idx] + a * DeltaT;
	pos_out[idx] = pos_in[idx] + DeltaT * (vel_out[idx] + 0.5f * DeltaT * a);


	if (y == height - 1)
	{
		int span = (width >> 2);
		int span2 = span + span;
		int span3 = span + span2;
		int span4 = width - 1;

		if (x == 0 || x == span || x == span2 || x == span3 || x == span4)
		{
			vel_out[idx] = vel_in[idx];
			pos_out[idx] = pos_in[idx];
		}
	}

}

__kernel
void cloth_position_local_modified(
	__global float4* pos_in, __global float4* pos_out,
	__global float4* vel_in, __global float4* vel_out,
	__global float4* force_in, __global float4* force_out,
	__local float4* local_data,
	float3 Gravity,
	float ParticleMass,
	float ParticleInvMass,
	float SpringK,
	float RestLengthHoriz,
	float RestLengthVert,
	float RestLengthDiag,
	float DeltaT,
	float DampingConst
	) {

	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int idx = x + width * y;
	float4 Fs, Fg, Fd, a, nexta;


	move_global_to_local(idx, pos_in, local_data);

	/*********************** Fs 계산 ****************************/
	Fs = getlocalFs(x, y, local_data, SpringK, RestLengthHoriz, RestLengthVert, RestLengthDiag, width, height);

	/*********************** Fg 계산 ****************************/
	Fg = ParticleMass * (float4)(Gravity, 0.0f);

	/*********************** Fd 계산 ****************************/
	Fd = -DampingConst * vel_in[idx];

	/*********************** 다음 힘 계산 ****************************/
	force_in[idx] = Fs + Fg + Fd;
	/*********************** 가속도 계산 ****************************/
	a = force_in[idx] * ParticleInvMass;

	/*********************** Fs 계산 ****************************/
	local_refresh(idx, local_data, pos_in, vel_in, DeltaT);
	Fs = getlocalFs(x, y, local_data, SpringK, RestLengthHoriz, RestLengthVert, RestLengthDiag, width, height);
	//Fs = getlocalnewFs(x, y, local_data, vel_in, DeltaT, SpringK, RestLengthHoriz, RestLengthVert, RestLengthDiag, width, height);

	/*********************** Fg 계산 ****************************/
	Fg = ParticleMass * (float4)(Gravity, 0.0f);

	/*********************** Fd 계산 ****************************/
	Fd = -DampingConst * (vel_in[idx] + a * DeltaT);

	/*********************** 다음 힘 계산 ****************************/
	force_out[idx] = Fs + Fg + Fd;
	nexta = force_out[idx] * ParticleInvMass;

	/*********************** First Order 계산 ****************************/
	vel_out[idx] = vel_in[idx] + 0.5f * (a + nexta)* DeltaT;
	pos_out[idx] = pos_in[idx] + 0.5f * (vel_in[idx] + vel_out[idx]) * DeltaT;


	if (y == height - 1)
	{
		int span = (width >> 2);
		int span2 = span + span;
		int span3 = span + span2;
		int span4 = width - 1;

		if (x == 0 || x == span || x == span2 || x == span3 || x == span4)
		{
			vel_out[idx] = vel_in[idx];
			pos_out[idx] = pos_in[idx];
		}
	}

}

/*
#define local_newpos(idx, gidx) (pos_in[idx] - vel_in[gidx]*DELTA_T)
inline float4 getlocalnewFs(int x, int y, __local float4* pos_in, __global float4* vel_in, float DELTA_T, float SpringK, float RestLengthHoriz, float RestLengthVert, float RestLengthDiag, float NUM_PARTICLES_X, float NUM_PARTICLES_Y)
{
int idx, gidx;
float4 diff;
float distance;
//계산하기 전 Fs는 Clear
float4 Fs = (float4)(0);

int i = local_direct(x, y, CENTER, NUM_PARTICLES_X, NUM_PARTICLES_Y);
int gi = direct(x, y, CENTER, NUM_PARTICLES_X, NUM_PARTICLES_Y);

idx = local_direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
gidx = direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
if (idx >= 0)
{
//Fs에 결과값을 더해준다.
// diff = r 구하기
diff = local_newpos(idx, gidx) - local_newpos(i, gi);
// distance = |r|
distance = fast_length(diff.xyz);

// diff = r/|r|
diff = (float4)(diff.xyz / distance, 0.0f);

// F = K *(|r| - R)*(r/|r|)
Fs += (SpringK * (distance - RestLengthHoriz)) * diff;
}
idx = local_direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
gidx = direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
if (idx >= 0)
{
//Fs에 결과값을 더해준다.
// diff = r 구하기
diff = local_newpos(idx, gidx) - local_newpos(i, gi);
// distance = |r|
distance = fast_length(diff.xyz);

// diff = r/|r|
diff = (float4)(diff.xyz / distance, 0.0f);

// F = K *(|r| - R)*(r/|r|)
Fs += (SpringK * (distance - RestLengthVert)) * diff;
}

idx = local_direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
gidx = direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
if (idx >= 0)
{
//Fs에 결과값을 더해준다.
// diff = r 구하기
diff = local_newpos(idx, gidx) - local_newpos(i, gi);
// distance = |r|
distance = fast_length(diff.xyz);
// diff = r/|r|
diff = (float4)(diff.xyz / distance, 0.0f);
// F = K *(|r| - R)*(r/|r|)
Fs += (SpringK * (distance - RestLengthHoriz)) * diff;
}
idx = local_direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
gidx = direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
if (idx >= 0)
{
//Fs에 결과값을 더해준다.
// diff = r 구하기s
diff = local_newpos(idx, gidx) - local_newpos(i, gi);
// distance = |r|
distance = fast_length(diff.xyz);
// diff = r/|r|
diff = (float4)(diff.xyz / distance, 0.0f);
// F = K *(|r| - R)*(r/|r|)
Fs += (SpringK * (distance - RestLengthVert)) * diff;
}

idx = local_direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
gidx = direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
if (idx >= 0)
{
//Fs에 결과값을 더해준다.
// diff = r 구하기
diff = local_newpos(idx, gidx) - local_newpos(i, gi);
// distance = |r|
distance = fast_length(diff.xyz);
// diff = r/|r|
diff = (float4)(diff.xyz / distance, 0.0f);
// F = K *(|r| - R)*(r/|r|)
Fs += (SpringK * (distance - RestLengthDiag)) * diff;
}
idx = local_direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
gidx = direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
if (idx >= 0)
{
//Fs에 결과값을 더해준다.
// diff = r 구하기
diff = local_newpos(idx, gidx) - local_newpos(i, gi);
// distance = |r|
distance = fast_length(diff.xyz);
// diff = r/|r|
diff = (float4)(diff.xyz / distance, 0.0f);
// F = K *(|r| - R)*(r/|r|)
Fs += (SpringK * (distance - RestLengthDiag)) * diff;
}
idx = local_direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
gidx = direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
if (idx >= 0)
{
//Fs에 결과값을 더해준다.
// diff = r 구하기
diff = local_newpos(idx, gidx) - local_newpos(i, gi);
// distance = |r|
distance = fast_length(diff.xyz);
// diff = r/|r|
diff = (float4)(diff.xyz / distance, 0.0f);
// F = K *(|r| - R)*(r/|r|)
Fs += (SpringK * (distance - RestLengthDiag)) * diff;
}
idx = local_direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
gidx = direct(x, y, LEFT, NUM_PARTICLES_X, NUM_PARTICLES_Y);
if (idx >= 0)
{
//Fs에 결과값을 더해준다.
// diff = r 구하기
diff = local_newpos(idx, gidx) - local_newpos(i, gi);
// distance = |r|
distance = fast_length(diff.xyz);
// diff = r/|r|
diff = (float4)(diff.xyz / distance, 0.0f);
// F = K *(|r| - R)*(r/|r|)
Fs += (SpringK * (distance - RestLengthDiag)) * diff;
}

return Fs;
}
*/
