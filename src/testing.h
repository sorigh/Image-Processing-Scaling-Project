//
// Created by Sorana on 5/18/2025.
//

#ifndef TESTING_H
#define TESTING_H


#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

double getPSNR(const Mat& im1, const Mat& im2);
Scalar getMSSIM(const Mat& im1, const Mat& im2);
double getRMSE(const Mat& im1, const Mat& im2);


#endif //TESTING_H
