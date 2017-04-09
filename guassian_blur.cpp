#include <opencv2/core.hpp>

#include "guassian_blur.h"

using namespace cv;
using namespace std;

Mat guassian_blur(Mat image)
{
	double g_curve[5][5] = { { 0.003765, 0.015019, 0.023792, 0.015019, 0.003765 },
							{ 0.015019, 0.059912, 0.094907, 0.059912, 0.015019 },
							{ 0.023792, 0.094907, 0.150342, 0.094907, 0.023792 },
							{ 0.003765, 0.015019, 0.023792, 0.015019, 0.003765 },
							{ 0.015019, 0.059912, 0.094907, 0.059912, 0.015019 } };


	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			double newBlue = 0.0;
			double newGreen = 0.0;
			double newRed = 0.0;

			Vec3b pixel = image.at<Vec3b>(row, col);
			double blue = pixel.val[0];
			double green = pixel.val[1];
			double red = pixel.val[2];
			// ^^^ this is how you get the color info

			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 5; j++) {

					// TODO implement this
					



				}
			}


		}
	}





	
	return image;
}

