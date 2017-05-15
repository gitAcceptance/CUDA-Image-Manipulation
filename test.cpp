#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "timer.h"

#include "gaussian_blur.h"
#include "gpu_functions.cu"

using namespace cv;
using namespace std;

float *h_filter__;

void make_filter(float** h_filter, int filterWidth) {

	if (filterWidth % 2 == 0) {
		cout << "STENCIL SIZE NEEDS TO BE ODD SCRUBERINO!";
		exit(1);
	}

	//now create the filter that they will use
	const int blurKernelWidth = filterWidth;
	const float blurKernelSigma = 2.;

	//*filterWidth = blurKernelWidth;

	//create and fill the filter we will convolve with
	*h_filter = new float[blurKernelWidth * blurKernelWidth];
	h_filter__ = *h_filter;

	float filterSum = 0.f; //for normalization

	for (int r = -blurKernelWidth / 2; r <= blurKernelWidth / 2; ++r) {
		for (int c = -blurKernelWidth / 2; c <= blurKernelWidth / 2; ++c) {
			float filterValue = expf(-(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
			(*h_filter)[(r + blurKernelWidth / 2) * blurKernelWidth + c + blurKernelWidth / 2] = filterValue;
			filterSum += filterValue;
		}
	}

	float normalizationFactor = 1.f / filterSum;

	for (int r = -blurKernelWidth / 2; r <= blurKernelWidth / 2; ++r) {
		for (int c = -blurKernelWidth / 2; c <= blurKernelWidth / 2; ++c) {
			(*h_filter)[(r + blurKernelWidth / 2) * blurKernelWidth + c + blurKernelWidth / 2] *= normalizationFactor;
		}
	}
}


int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	Mat input_image;
	input_image = imread(argv[1], IMREAD_COLOR); // Read the file

	if (input_image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	//make sure the context initializes ok
	checkCudaErrors(cudaFree(0));

	float *h_filter;
	int filterWidth = 5;

	make_filter(&h_filter, filterWidth); // making a filter of size 5

	cv::Mat imageInputRGBA;
	cv::Mat imageOutputRGBA;

	uchar4 *h_inputImageRGBA, *d_inputImageRGBA;
	uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
	unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;

	cv::cvtColor(input_image, imageInputRGBA, CV_BGR2RGBA);

	//allocate memory for the output
	imageOutputRGBA.create(input_image.rows, input_image.cols, CV_8UC4);

	h_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
	h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);






	allocateMemoryAndCopyToGPU(input_image.rows, input_image.cols, h_filter, filterWidth);
	GpuTimer timer;
	timer.Start();
	//call my code
	gpu_gaussian_blur(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, input_image.rows, input_image.cols,
		d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);
	timer.Stop();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());



	/*
	Mat new_image = gaussian_blur(image);
	


	namedWindow("Original Image", WINDOW_AUTOSIZE); // Create a window for display.
	namedWindow("Blurred Image", WINDOW_AUTOSIZE);

	//imshow("Display window", image); // Show our image inside it.
	imshow("Original Image", image);
	imshow("Blurred Image", new_image);

	imwrite("BlurredFossum.jpg", new_image);
	

	waitKey(0); // Wait for a keystroke in the window
	*/

	delete[] h_filter__;
	return 0;
}

