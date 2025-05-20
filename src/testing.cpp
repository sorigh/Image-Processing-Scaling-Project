//
// Created by Sorana on 5/18/2025.
//

#include "testing.h"


// calculates peak signal-to-noise ratio between two grayscale images
// how different the second image is from the original based on mse (mean squared error)
// shows how much noise/distortion there is
//
// higher PSNR value -> higher image quality (less distortion).
// if the images are identical, PSNR = infinity.
//
// @param im1 - original image
// @param im2 - compared image
// @return PSNR value in decibels (dB)
//
double getPSNR(const Mat& im1, const Mat& im2) {
    Mat s1;
    absdiff(im1, im2, s1);// |im1 - im2|
    s1.convertTo(s1, CV_32F);//to float
    s1 = s1.mul(s1);// square each element

    Scalar s = sum(s1);// sum all elements (squared errors)

    double sse = s.val[0];// sum of squared errors
    if (sse <= 1e-10) // if sse is small, images are identical
        return std::numeric_limits<double>::infinity();
    //else
    double mse = sse / (double)(im1.total());
    double psnr = 10.0 * log10((255 * 255) / mse); //formula
    return psnr;
}

// calculates the ssim (structural similarity index) between two grayscale images
// ssim is another way to measure image quality — but it also considers structure and contrast
// it's supposed to be more "human-like" in terms of how it judges image similarity
// returns a value between -1 and 1, where 1 means they look the same


// Computes the Mean Structural Similarity Index (SSIM) between two grayscale images.
// SSIM is designed to assess perceived visual quality by measuring similarity in luminance,
// contrast, and structure. It is more aligned with human visual perception than PSNR.
//
// The function applies Gaussian blurring to compute local statistics (mean, variance, covariance)
// and then combines them into an SSIM map. The mean of this map is returned.
//
// @param i1 - original image (CV_8UC1 expected)
// @param i2 - compared image (CV_8UC1 expected)
// @return Scalar containing mean SSIM (one value per channel, but only one used here)
//
Scalar getMSSIM(const Mat& im1, const Mat& im2) {
    const double C1 = 6.5025, C2 = 58.5225; // constants to avoid division by zero
    Mat i1, i2;

    // float versions of the images
    im1.convertTo(i1, CV_32F);
    im2.convertTo(i2, CV_32F);

    Mat i2_2 = i2.mul(i2);     // i2 squared
    Mat i1_2 = i1.mul(i1);     // i1 squared
    Mat i1_i2 = i1.mul(i2);    // i1 * i2

    // blurred versions (mean values)
    Mat mu1, mu2;
    GaussianBlur(i1, mu1, Size(11, 11), 1.5);
    GaussianBlur(i2, mu2, Size(11, 11), 1.5);

    //square means and multiply
    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);


    // calculate variance and covariance

    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(i1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(i2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(i1_i2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;


    // apply the ssim formula
    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2); //numerator

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2); //denominator

    Mat ssim_map;
    divide(t3, t1, ssim_map); // pixel-wise ssim
    Scalar mssim = mean(ssim_map);// average over the whole image
    return mssim;
}
// calculates the root mean square error between two grayscale images
// this just measures the average pixel difference (similar to mse but in original units)
// A lower RMSE -> higher similarity.
//
// @param I1 - original image
// @param I2 - compared image
// @return RMSE value (0 = perfect match)
//
double getRMSE(const Mat& im1, const Mat& im2) {
    Mat diff;
    absdiff(im1, im2, diff); // pixel-wise difference
    diff.convertTo(diff, CV_32F);// convert to float
    diff = diff.mul(diff);// square each difference


    Scalar sumSq = sum(diff); // sum of squared differences
    double rmse = sqrt(sumSq[0] / (im2.total())); // take square root of average
    return rmse;
}



