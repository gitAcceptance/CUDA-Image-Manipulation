#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include "gaussian_blur.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	Mat image;
	image = imread(argv[1], IMREAD_COLOR); // Read the file

	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}


	Mat new_image = gaussian_blur(image);
	new_image = gaussian_blur(new_image);
	new_image = gaussian_blur(new_image);
	new_image = gaussian_blur(new_image);
	new_image = gaussian_blur(new_image);
	new_image = gaussian_blur(new_image);


	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window", image); // Show our image inside it.
	imshow("Display window", new_image);

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}

