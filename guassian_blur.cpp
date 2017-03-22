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
			float newValue = -1;

			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 5; j++) {




				}
			}


		}
	}





	
	return image;
}

