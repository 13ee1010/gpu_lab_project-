#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

// Read value from global array a, return 0 if outside image
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void sobelKernel(__read_only image2d_t d_input, __global float* d_output) {
	float maximum = 0.0f;
	int structure_element[3][3] = { {1,1,1},
									 {1,1,1},
									 {1,1,1}
	};
	int i = get_global_id(0);
	int j = get_global_id(1);
	
	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{    
			const float pxVal = read_imagef(d_input, sampler, (int2)(x+i, y+j)).x;
		//	printf("pixel value for co-ordinates %d, %d is %f\n", x + i, y + j, pxVal);
			if (structure_element[x + 1][y + 1] == 1)
			{
				maximum = max(maximum, pxVal);
			}
			

		}
	}
	d_output[countX*j + i] = maximum;
}

