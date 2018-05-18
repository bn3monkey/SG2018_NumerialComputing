__kernel void reduction_global
(__global float* input, __global float* output)
{
	int col = get_local_id(0);
	int row = get_local_id(1);

	int C = get_local_size(0);
	int R = get_local_size(1);

	int target = get_group_id(1) * get_num_groups(0) + get_group_id(0);
	int mem_start = target * C * R;

	for (int i = R >> 1; i > 0; i >>= 1)
	{
		if (row < i)
		{
			input[mem_start + row * C + col] += input[mem_start + (row + i) * C + col];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (row == 0)
	{
		for (int i = C >> 1; i > 0; i >>= 1)
		{
			if (col < i)
			{
				input[mem_start + col] += input[mem_start + col + i];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
	if (row == 0 && col == 0)
		output[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = input[mem_start];
}

__kernel void reduction_local
(__global float* input, __global float* output, __local float* temp)
{
	int col = get_local_id(0);
	int row = get_local_id(1);

	int C = get_local_size(0);
	int R = get_local_size(1);

	int target = get_group_id(1) * get_num_groups(0) + get_group_id(0);
	int mem_start = target * C * R;

	temp[row * C + col] = input[mem_start + row * C + col];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = R >> 1; i > 0; i >>= 1)
	{
		if (row < i)
		{
			temp[row * C + col] += temp[(row + i) * C + col];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (row == 0)
	{
		for (int i = C >> 1; i > 0; i >>= 1)
		{
			if (col < i)
			{
				temp[col] += temp[col + i];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
	if (row == 0 && col == 0)
		output[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = temp[0];
}