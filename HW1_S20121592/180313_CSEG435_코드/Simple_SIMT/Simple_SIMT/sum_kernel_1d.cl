__kernel void reduction_global
(__global float* input, __global float* output)
{
	int lid = get_local_id(0);
	int id = get_global_id(0);
	int size = get_local_size(0) >> 1;
	
	for (int i = size; i > 0; i >>= 1)
	{
		//if (get_global_id(0) == 0)
		//	printf("temp[%d] = %f temp[%d] = %f\n", id, input[id], id + i, input[id + i]);

		if (lid < i)
		{
			input[id] += input[id + i];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	if (lid == 0)
		output[get_group_id(0)] = input[id];

}

__kernel void reduction_local
(__global float* input, __global float* output, __local float* temp)
{
	int lid = get_local_id(0);
	int group_size = get_local_size(0) >> 1;


	temp[lid] = input[get_global_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);


	for (int i = group_size; i > 0; i >>= 1)
	{
		//if (lid == 0 && get_group_id(0) == 0)
		//	printf("temp[%d] = %f temp[%d] = %f\n", lid, temp[lid], lid+i, temp[lid+i]);

		if (lid < i)
			temp[lid] += temp[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (lid == 0)
	{
		output[get_group_id(0)] = temp[0];
	}
	
}