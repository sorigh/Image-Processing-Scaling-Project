#ifndef INTP_H
#define INTP_H
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat nearestNeighbor(const Mat& input, double scaleX, double scaleY);
Mat bilinear(const Mat& input, double scaleX, double scaleY);
Mat bicubicCustom(const Mat& input, double scaleX, double scaleY, float a = -0.5f);
Mat lanczos(const Mat& input, double scaleX, double scaleY, int a = 3);

#endif