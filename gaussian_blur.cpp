#include <opencv2/core.hpp>

#include "gaussian_blur.h"

using namespace cv;
using namespace std;

Mat gaussian_blur(Mat image)
{
	double g_curve[5][5] = { { 0.003765, 0.015019, 0.023792, 0.015019, 0.003765 },
							{ 0.015019, 0.059912, 0.094907, 0.059912, 0.015019 },
							{ 0.023792, 0.094907, 0.150342, 0.094907, 0.023792 },
							{ 0.003765, 0.015019, 0.023792, 0.015019, 0.003765 },
							{ 0.015019, 0.059912, 0.094907, 0.059912, 0.015019 } };

	//Mat newImage(Size(image.cols , image.rows), image.type);

	Mat newImage = image.clone();

	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			double newBlue = 0.0;
			double newGreen = 0.0;
			double newRed = 0.0;

			// Vec3b pixel = image.at<Vec3b>(row, col);
			// double blue = pixel.val[0];
			// double green = pixel.val[1];
			// double red = pixel.val[2];
			// ^^^ this is how you get the color info

			double toBeSummedBlue[25];
			double toBeSummedGreen[25];
			double toBeSummedRed[25];
			

			int tempRow = row - 2;
			int tempCol = col - 2;
			int count = 0;

			// TODO get rid of literal 5
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 5; j++) {

					// TODO fix this bit so it uses the edge values instead of going out of bounds
					if (tempRow + i >= image.rows || tempCol + j >= image.cols) {
						continue;
					}
					if (tempRow + i < 0 || tempCol + j < 0) {
						continue;
					}

					Vec3b pixel = image.at<Vec3b>(tempRow + i, tempCol + j);

					toBeSummedBlue[count]  = pixel.val[0] * g_curve[i][j];
					toBeSummedGreen[count] = pixel.val[1] * g_curve[i][j];
					toBeSummedRed[count]   = pixel.val[2] * g_curve[i][j];

					count++;
				}
			}

			// TODO get rid of literal 25
			for (int i = 0; i < 25; i++) {
				newBlue += toBeSummedBlue[i];
				newGreen += toBeSummedGreen[i];
				newRed += toBeSummedRed[i];
			}

			// TODO take those new values and put them in a pixel
			Vec3b newPixel = Vec3b(newBlue, newGreen, newRed);
			newImage.at<Vec3b>(row, col) = newPixel;
		}
	}

	// TODO return the new image



	
	return newImage;
}

