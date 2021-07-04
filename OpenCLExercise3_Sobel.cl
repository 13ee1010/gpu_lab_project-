#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif


// Read value from global array a, return 0 if outside image
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

float getValueImage(__read_only image2d_t a, int i, int j) {
	 return read_imagef(a, sampler, (int2) { i, j }).x;
}

// Read value from global array a, return 0 if outside image
float getValueGlobal(__global const float* a, size_t countX, size_t countY, int i, int j) {
	if (i < 0 || i >= countX || j < 0 || j >= countY)
		return 0;
	else
		return a[countX * j + i];
}


__kernel void dilation(__read_only image2d_t d_input, __global float* d_outputDilation) {
	
	int i = get_global_id(0);
	int j = get_global_id(1);
	
	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	float maximum = 0.0f;
	int structure_element[3][3] = { {0,1,0},
									 {1,1,1},
									 {0,1,0}
	};
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			//const float pxVal = read_imagef(d_input, sampler, (int2)(x+i, y+j)).x;
			const float pxVal = getValueImage(d_input, x + i, y + j);
			//	printf("pixel value for co-ordinates %d, %d is %f\n", x + i, y + j, pxVal);
			if (structure_element[x + 1][y + 1] == 1)
			{
				if (maximum > pxVal)
					maximum = maximum;
				else
					maximum = pxVal;
			}


		}
	}
	d_outputDilation[countX * j + i] = maximum;
	
}

__kernel void erosion(__read_only image2d_t d_input, __global float* d_outputErosion) {
	
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);
	
	float minimum = 1.0f;
	int structure_element[3][3] = { {0,1,0},
									 {1,1,1},
									 {0,1,0}
	};
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			//const float pxVal = read_imagef(d_input, sampler, (int2)(x+i, y+j)).x;
			const float pxVal = getValueImage(d_input, x + i, y + j);
			//	printf("pixel value for co-ordinates %d, %d is %f\n", x + i, y + j, pxVal);
			if (structure_element[x + 1][y + 1] == 1)
			{
				if (minimum < pxVal)
					minimum = minimum;
				else
					minimum = pxVal;
			}


		}
	}
	d_outputErosion[countX * j + i] = minimum;
}

__kernel void opening(__read_only image2d_t d_input, __global float* d_outputOpening, __global float* d_temp1) {

	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	float minimum = 1.0f;
	float maximum = 0.0f;
	int structure_element[3][3] = { {0,1,0},
									 {1,1,1},
									 {0,1,0}
	};
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			//const float pxVal = read_imagef(d_input, sampler, (int2)(x+i, y+j)).x;
			const float pxVal = getValueImage(d_input, x + i, y + j);
			//	printf("pixel value for co-ordinates %d, %d is %f\n", x + i, y + j, pxVal);
			if (structure_element[x + 1][y + 1] == 1)
			{
				if (minimum < pxVal)
					minimum = minimum;
				else
					minimum = pxVal;
			}


		}
	}
	d_temp1[countX * j + i] = minimum;
	barrier(CLK_GLOBAL_MEM_FENCE);
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			//const float pxVal = read_imagef(d_input, sampler, (int2)(x+i, y+j)).x;
			const float pxVal2 = getValueGlobal(d_temp1, countX, countY, x + i, y + j);
			//	printf("pixel value for co-ordinates %d, %d is %f\n", x + i, y + j, pxVal);
			if (structure_element[x + 1][y + 1] == 1)
			{
				if (maximum > pxVal2)
					maximum = maximum;
				else
					maximum = pxVal2;
			}


		}
	}
	d_outputOpening[countX * j + i] = maximum;

	
}

__kernel void closing(__read_only image2d_t d_input, __global float* d_outputClosing, __global float* d_temp2) {

	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	float minimum = 1.0f;
	float maximum = 0.0f;
	int structure_element[3][3] = { {0,1,0},
									 {1,1,1},
									 {0,1,0}
	};

	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			//const float pxVal = read_imagef(d_input, sampler, (int2)(x+i, y+j)).x;
			const float pxVal2 = getValueImage(d_input, x + i, y + j);
			//	printf("pixel value for co-ordinates %d, %d is %f\n", x + i, y + j, pxVal);
			if (structure_element[x + 1][y + 1] == 1)
			{
				if (maximum > pxVal2)
					maximum = maximum;
				else
					maximum = pxVal2;
			}


		}
	}
	d_temp2[countX * j + i] = maximum;
	barrier(CLK_GLOBAL_MEM_FENCE);
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			//const float pxVal = read_imagef(d_input, sampler, (int2)(x+i, y+j)).x;
			const float pxVal = getValueGlobal(d_temp2, countX, countY, x + i, y + j);
			//	printf("pixel value for co-ordinates %d, %d is %f\n", x + i, y + j, pxVal);
			if (structure_element[x + 1][y + 1] == 1)
			{
				if (minimum < pxVal)
					minimum = minimum;
				else
					minimum = pxVal;
			}


		}
	}
	d_outputClosing[countX * j + i] = minimum;

}