#include <iostream>
#include "OpenCVEx.h"
using namespace std;

void HelloWorld()
{
	Mat image(1000, 1000, CV_8UC3, cv::Scalar(255, 0, 0));
	imshow("image", image);
	waitKey(0);
}