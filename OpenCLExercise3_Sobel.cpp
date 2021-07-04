//////////////////////////////////////////////////////////////////////////////
// OpenCL exercise 3: Sobel filter
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

//#include <boost/lexical_cast.hpp>

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
float maximum = 0.0f;
int structure_element[3][3] = { {0,1,0},
								 {1,1,1},
								 {0,1,0}
};

float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j) {
	if (i < 0 || (size_t)i >= countX || j < 0 || (size_t)j >= countY)
		return 0;
	else
		return a[j*countX + i];
}
void dilation(const std::vector<float>& h_input, std::vector<float>& h_outputDilationCpu, std::size_t countX, std::size_t countY) {
	float maximum = 0.0f;
	for (int i = 0; i < (int)countX; i++)
	{
		for (int j = 0; j < (int)countY; j++)
		{
			maximum = 0.0f;
			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{   
					float pixelValue = getValueGlobal(h_input, countX, countY, x + i, y + j);
					if (structure_element[x + 1][y + 1] == 1)
					{
						
						if (maximum > pixelValue)
							maximum = maximum;
						else
							maximum = pixelValue;
					}
					
				}
			}
			h_outputDilationCpu[j * countX + i ] = maximum;
		}
	}
}

void erosion(const std::vector<float>& h_input, std::vector<float>& h_outputErosionCpu, std::size_t countX, std::size_t countY) {
	float minimum = 1.0f;
	for (int i = 0; i < (int)countX; i++)
	{
		for (int j = 0; j < (int)countY; j++)
		{
			minimum = 1.0f;
			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{
					float pixelValue = getValueGlobal(h_input, countX, countY, x + i, y + j);
					if (structure_element[x + 1][y + 1] == 1)
					{

						if (minimum < pixelValue)
							minimum = minimum;
						else
							minimum = pixelValue;
					}

				}
			}
			h_outputErosionCpu[j * countX + i] = minimum;
		}
	}
}

void opening(const std::vector<float>& h_input, std::vector<float>& h_outputOpeningCpu, std::size_t countX, std::size_t countY)
{
	std::size_t count1 = countX * countY;
	std::vector<float> h_temp(count1);
	erosion(h_input, h_temp, countX, countY);
	dilation(h_temp, h_outputOpeningCpu, countX, countY);
}

void closing(const std::vector<float>& h_input, std::vector<float>& h_outputClosingCpu, std::size_t countX, std::size_t countY)
{
	std::size_t count1 = countX * countY;
	std::vector<float> h_temp1(count1);
	dilation(h_input, h_temp1, countX, countY);
	erosion(h_temp1, h_outputClosingCpu, countX, countY);
}

//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	// Create a context
	//cl::Context context(CL_DEVICE_TYPE_GPU);
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[platformId](), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);


	// Get a device of the context
	int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
	std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
	ASSERT(deviceNr > 0);
	ASSERT((size_t)deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "D:/UNIVERSITY_OF_STUTTGART/COURSES_MATERIAL/SEM4/GPU_LAB/Opencl-ex1/Opencl-ex1/src/OpenCLExercise3_Sobel.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Declare some values /// can increase the size -- SUMIT 
	std::size_t wgSizeX = 20; // Number of work items per work group in X direction
	std::size_t wgSizeY = 20;
	std::size_t countX = wgSizeX; // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = wgSizeY;
	//countX *= 3; countY *= 3;
	std::size_t count = countX * countY; // Overall number of elements
	std::size_t size = count * sizeof(float); // Size of data in bytes

	// Allocate space for output data from CPU and GPU on the host
	std::vector<float> h_input(count);
	std::vector<float> h_outputDilationCpu(count);
	std::vector<float> h_outputErosionCpu(count);

	std::vector<float> h_outputOpeningCpu(count);
	std::vector<float> h_outputClosingCpu(count);

	std::vector<float> h_outputDilationGpu(count);
	std::vector<float> h_outputErosionGpu(count);

	std::vector<float> h_outputOpeningGpu(count);
	std::vector<float> h_outputClosingGpu(count);
	
	std::vector<float> h_temp1(count);
	std::vector<float> h_temp2(count);

	// Allocate space for input and output data on the device
	cl::Buffer d_input(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputDilation(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputErosion(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputOpening(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputClosing(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_temp1(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_temp2(context, CL_MEM_READ_WRITE, size);
	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
	memset(h_outputDilationCpu.data(), 255, size);
	memset(h_outputErosionCpu.data(), 255, size);

	memset(h_outputOpeningCpu.data(), 255, size);
	memset(h_outputClosingCpu.data(), 255, size);

    memset(h_outputDilationGpu.data(), 255, size);
	memset(h_outputErosionGpu.data(), 255, size);

	memset(h_outputOpeningGpu.data(), 255, size);
	memset(h_outputClosingGpu.data(), 255, size);
    
	memset(h_temp2.data(), 255, size);
	memset(h_temp1.data(), 255, size);


	//TODO: GPU
	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(d_outputDilation, true, 0, size, h_outputDilationGpu.data());
	queue.enqueueWriteBuffer(d_outputErosion, true, 0, size, h_outputErosionGpu.data());
	queue.enqueueWriteBuffer(d_outputOpening, true, 0, size, h_outputOpeningGpu.data());
	queue.enqueueWriteBuffer(d_outputClosing, true, 0, size, h_outputClosingGpu.data());
	queue.enqueueWriteBuffer(d_temp1, true, 0, size, h_temp1.data());
	queue.enqueueWriteBuffer(d_temp2, true, 0, size, h_temp2.data());


	//////// Load input data ////////////////////////////////
	// Use random input data
	/*
	for (int i = 0; i < count; i++)
		h_input[i] = (rand() % 100) / 5.0f - 10.0f;
	*/
	// Use an image (Valve.pgm) as input data
	{
		std::vector<float> inputData;
		std::size_t inputWidth, inputHeight;
		Core::readImagePGM("D:/UNIVERSITY_OF_STUTTGART/COURSES_MATERIAL/SEM4/GPU_LAB/Opencl-ex1/Opencl-ex1/data.pgm", inputData, inputWidth, inputHeight);
		for (size_t j = 0; j < countY; j++) {
			for (size_t i = 0; i < countX; i++) {
				h_input[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
			}
		}
	}

	// Do calculation on the host side
	Core::TimeSpan cpuStart = Core::getCurrentTime();
	dilation(h_input, h_outputDilationCpu, countX, countY);
	erosion(h_input, h_outputErosionCpu, countX, countY);
	opening(h_input, h_outputOpeningCpu, countX, countY);
	closing(h_input, h_outputClosingCpu, countX, countY);
    Core::TimeSpan cpuEnd = Core::getCurrentTime();

	//////// Store CPU output image ///////////////////////////////////
	Core::writeImagePGM("output_dilation_cpu_sumit.pgm", h_outputDilationCpu, countX, countY);
	Core::writeImagePGM("output_erosion_cpu_sumit.pgm", h_outputErosionCpu, countX, countY);
	Core::writeImagePGM("output_opening_cpu_sumit.pgm", h_outputOpeningCpu, countX, countY);
	Core::writeImagePGM("output_closing_cpu_sumit.pgm", h_outputClosingCpu, countX, countY);
	std::cout << std::endl;
	// Iterate over all implementations (task 1 - 3)

	std::cout << "Implementation #" << ":" << std::endl;

	// Reinitialize output memory to 0xff
	memset(h_outputDilationGpu.data(), 255, size);
	memset(h_outputErosionGpu.data(), 255, size);
	memset(h_outputOpeningGpu.data(), 255, size);
	memset(h_outputClosingGpu.data(), 255, size);
	memset(h_temp1.data(), 255, size);
	memset(h_temp1.data(), 255, size);
	//TODO: GPU
	queue.enqueueWriteBuffer(d_outputDilation, true, 0, size, h_outputDilationGpu.data());
	queue.enqueueWriteBuffer(d_outputErosion, true, 0, size, h_outputErosionGpu.data());
	queue.enqueueWriteBuffer(d_outputOpening, true, 0, size, h_outputOpeningGpu.data());
	queue.enqueueWriteBuffer(d_outputClosing, true, 0, size, h_outputClosingGpu.data());
	// Copy input data to device
	cl::Event copy1;
	cl::Image2D image;

	image = cl::Image2D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);
	cl::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;
	cl::size_t<3> region;
	region[0] = countX;
	region[1] = countY;
	region[2] = 1;
	queue.enqueueWriteImage(image, true, origin, region, countX * sizeof(float), 0, h_input.data(), NULL, &copy1);


	// Create a kernel object
	std::string kernelName1 = "dilation";
	std::string kernelName2 = "erosion";
	std::string kernelName3 = "opening";
	std::string kernelName4 = "closing";
	cl::Kernel dilation(program, kernelName1.c_str());
	cl::Kernel erosion(program, kernelName2.c_str());
	cl::Kernel opening(program, kernelName3.c_str());
	cl::Kernel closing(program, kernelName4.c_str());

	// Launch kernel on the device
	cl::Event execution[4];

	dilation.setArg<cl::Image2D>(0, image);
	dilation.setArg<cl::Buffer>(1, d_outputDilation);
    queue.enqueueNDRangeKernel(dilation, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &execution[0]);

	erosion.setArg<cl::Image2D>(0, image);
	erosion.setArg<cl::Buffer>(1, d_outputErosion);
	queue.enqueueNDRangeKernel(erosion, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &execution[1]);

	opening.setArg<cl::Image2D>(0, image);
	opening.setArg<cl::Buffer>(1, d_outputOpening);
	opening.setArg<cl::Buffer>(2, d_temp1);
	queue.enqueueNDRangeKernel(opening, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &execution[2]);

	closing.setArg<cl::Image2D>(0, image);
	closing.setArg<cl::Buffer>(1, d_outputClosing);
	closing.setArg<cl::Buffer>(2, d_temp2);
	queue.enqueueNDRangeKernel(closing, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &execution[3]);

	// Copy output data back to host
	cl::Event copy2[4];
	queue.enqueueReadBuffer(d_outputDilation, true, 0, size, h_outputDilationGpu.data(), NULL, &copy2[0]);
	queue.enqueueReadBuffer(d_outputErosion, true, 0, size, h_outputErosionGpu.data(), NULL, &copy2[1]);
	queue.enqueueReadBuffer(d_outputOpening, true, 0, size, h_outputOpeningGpu.data(), NULL, &copy2[2]);
	queue.enqueueReadBuffer(d_outputClosing, true, 0, size, h_outputClosingGpu.data(), NULL, &copy2[3]);

	// Print performance data
	Core::TimeSpan cpuTime = cpuEnd - cpuStart;

	Core::TimeSpan gpuTime = Core::TimeSpan::fromSeconds(0);
	for (std::size_t i = 0; i < sizeof(execution) / sizeof(*execution); i++)
		gpuTime = gpuTime + OpenCL::getElapsedTime(execution[i]);

	Core::TimeSpan copytime2 = Core::TimeSpan::fromSeconds(0);
    for (std::size_t i = 0; i < sizeof(copy2) / sizeof(*copy2); i++)
		copytime2 = copytime2 + OpenCL::getElapsedTime(copy2[i]);

	Core::TimeSpan copyTime = OpenCL::getElapsedTime(copy1) + copytime2;
	Core::TimeSpan overallGpuTime = gpuTime + copyTime;
	std::cout << "CPU Time: " << cpuTime.toString() << ", " << (count / cpuTime.getSeconds() / 1e6) << " MPixel/s" << std::endl;;
	std::cout << "Memory copy Time: " << copyTime.toString() << std::endl;
	std::cout << "GPU Time w/o memory copy: " << gpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / gpuTime.getSeconds()) << ", " << (count / gpuTime.getSeconds() / 1e6) << " MPixel/s)" << std::endl;
	std::cout << "GPU Time with memory copy: " << overallGpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / overallGpuTime.getSeconds()) << ", " << (count / overallGpuTime.getSeconds() / 1e6) << " MPixel/s)" << std::endl;

	//////// Store GPU output image ///////////////////////////////////
	Core::writeImagePGM("output_dilation_gpu_sumit_data.pgm", h_outputDilationGpu, countX, countY);
	Core::writeImagePGM("output_erosion_gpu_sumit_data.pgm", h_outputErosionGpu, countX, countY);
	Core::writeImagePGM("output_opening_gpu_sumit_data.pgm", h_outputOpeningGpu, countX, countY);
	Core::writeImagePGM("output_closing_gpu_sumit_data.pgm", h_outputClosingGpu, countX, countY);
/*
	// Check whether results are correct
	std::size_t errorCount = 0;
	for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-direction
		for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-direction
			size_t index = i + j * countX;
			// Allow small differences between CPU and GPU results (due to different rounding behavior)
			if (!(std::abs(h_outputCpu[index] - h_outputGpu[index]) <= 1e-5)) {
				if (errorCount < 15)
				{
				std::cout << "input " << h_input[index] << " Result for " << i << "," << j << " is incorrect: GPU value is " << h_outputGpu[index] << ", CPU value is " << h_outputCpu[index] << std::endl;
				}
				else if (errorCount == 15)
					std::cout << "..." << std::endl;
				errorCount++;
			}
		}
	}
	if (errorCount != 0) {
		std::cout << "Found " << errorCount << " incorrect results" << std::endl;
		return 1;
	}

	std::cout << std::endl;


	std::cout << "Success" << std::endl;
*/
	return 0;
}
