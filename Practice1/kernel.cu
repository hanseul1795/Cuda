
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

#define ARRAY_SIZE 200
#define ARRAY_BYTES ARRAY_SIZE * sizeof(float)

__global__ void CalculateSquare(float* p_out, float* p_in)
{
	int index = threadIdx.x;
	float valueToSuqare = p_in[index];
	p_out[index] = valueToSuqare * valueToSuqare;
}

int main()
{
	float in[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		in[i] = static_cast<float>(i);
	}
	float out[ARRAY_SIZE];

	float* GPU_in;
	float* GPU_out;

	//Memory allocation in GPU
	cudaMalloc((void **)&GPU_in, ARRAY_BYTES);
	cudaMalloc((void **)&GPU_out, ARRAY_BYTES);

	//copy(send) result array to GPU
	cudaMemcpy(GPU_in, in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	//launch the kernal(the __global__ implementation above)
	CalculateSquare<<<1,ARRAY_SIZE>>>(GPU_out, GPU_in);

	//COPY calculated data from GPU to cpu
	cudaMemcpy(out, GPU_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		std::cout << out[i] << "     ";
		if (i % 4 == 0)
			std::cout << std::endl;
	}
	//DON'T FORGET TO FREE MEMORY
	cudaFree(GPU_in);
	cudaFree(GPU_out);
	system("Pause");

	return 0;
}