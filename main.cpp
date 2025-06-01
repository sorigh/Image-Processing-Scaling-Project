#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/interpolation.h"
#include "src/testing.h"
#include <fstream>

using namespace std;
using namespace cv;
using namespace chrono;

void saveImage(const Mat& img, const String& windowName) {
    imwrite("../output/" + windowName + ".png", img);
    cout << "Saved image: " << windowName << ".png" << endl;
}

Mat getImage(String name) {
    Mat image = imread("../images/" + name, IMREAD_GRAYSCALE);
    imshow(name, image);
    moveWindow(name, 0, 0);
    return image;
}

// logging for console, txt file and csv
void logResults(const string& imageName, const string& algoName, double fx, const Mat& scaled, const Mat& original,
                long long duration, ofstream& resultFile, ofstream& csvFile) {

    // resize scaled image back to original size (for comparison)
    Mat restored;
    resize(scaled, restored, original.size(), 0, 0, INTER_LINEAR);

    // quality metrics
    // peak signal to noise ratio -> amount of dif between original and processed img
    double psnr = getPSNR(original, restored);

    //structural similarity index -> visual similarity comparing brightness, contrast, structure
    Scalar ssim = getMSSIM(original, restored);

    // root mean square error -> avg error between pixels
    double rmse = getRMSE(original, restored);

    string outName = imageName + "_" + algoName + "_x" + to_string(fx).substr(0, 3);
    imwrite("../output/" + outName + ".png", scaled);

    string log = "Algoritm: " + algoName +
                 " | Time: " + to_string(duration) + " ms" +
                 " | PSNR: " + to_string(psnr) +
                 " | SSIM: " + to_string(ssim[0]) +
                 " | RMSE: " + to_string(rmse) + "\n";

    cout << log;
    resultFile << log;
    csvFile << imageName << ",x" << fx << "," << algoName << "," << duration << ","
             << psnr << "," << ssim[0] << "," << rmse << "\n";
}



void testInterpolationOnImage(const Mat& original, const string& imageName, double fx, double fy) {
    cout << "\n=== Testing on: " << imageName << " | Scale: " << fx << " x " << fy << " ===\n";

    // files
    ofstream resultFile("../output/doc/results.txt", ios::app);
    ofstream csvFile("../output/doc/results.csv", ios::app);

    resultFile << "\n=== Testing on: " << imageName << " | Scale: " << fx << " x " << fy << " ===\n";

    vector<pair<string, function<Mat(const Mat&, double, double)>>> basicAlgorithms = {
        {"Nearest", nearestNeighbor},
        {"Bilinear", bilinear}
    };

    //iterate & apply each basic alg and log results
    for (const auto& alg : basicAlgorithms) {
        //time
        auto start = high_resolution_clock::now();
        Mat scaled = alg.second(original, fx, fy);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();

        logResults(imageName, alg.first, fx, scaled, original, duration, resultFile, csvFile);
    }

    // change param a for bicubic interpolation
    for (double a : {-0.5, -0.75, -1.0}) {
        auto start = high_resolution_clock::now();
        Mat scaled = bicubicCustom(original, fx, fy, a);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();

        string algoName = "Bicubic (a=" + to_string(a).substr(0, 4) + ")";
        logResults(imageName, algoName, fx, scaled, original, duration, resultFile, csvFile);
    }

    //change param a for lanczos interpolation
    for (int a : {2, 3, 4}) {
        auto start = high_resolution_clock::now();
        Mat scaled = lanczos(original, fx, fy, a);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();

        string algoName = "Lanczos (a=" + to_string(a) + ")";
        logResults(imageName, algoName, fx, scaled, original, duration, resultFile, csvFile);
    }

    // ---------------------
    // compare with built in interpolation methods
    // ---------------------
    vector<pair<string, int>> opencvInterpolations = {
        {"OpenCV Nearest", INTER_NEAREST},
        {"OpenCV Bilinear", INTER_LINEAR},
        {"OpenCV Bicubic", INTER_CUBIC},
        {"OpenCV Lanczos", INTER_LANCZOS4}
    };

    for (const auto& interp : opencvInterpolations) {
        auto start = high_resolution_clock::now();
        Mat scaled;
        resize(original, scaled, Size(), fx, fy, interp.second);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();

        logResults(imageName, interp.first, fx, scaled, original, duration, resultFile, csvFile);
    }

    resultFile.close();
    csvFile.close();
}

void testMain() {
    // list of image files
    vector<string> imageNames = {"shell.png", "binaryflower.png", "balloons.bmp"};

    // scale
    vector<pair<double, double>> scales = {
        {2.0, 2.0}, // upscale
        {0.5, 0.5}, // downscale
        {1.5, 0.75} // non uniform scale
    };

    // init CSV with header
    ofstream csvHeader("../output/doc/results.csv");
    csvHeader << "Image,Scale,Algorithm,Time (ms),PSNR,SSIM,RMSE\n";
    csvHeader.close();

    // loop over images
    for (const auto& imageName : imageNames) {
        Mat image = imread("../images/" + imageName, IMREAD_GRAYSCALE);
        if (image.empty()) {
            cout << "Could not open or find the image: " << imageName << endl;
            continue;
        }

        // for each scale apply interpolation
        for (const auto& scale : scales) {
            testInterpolationOnImage(image, imageName.substr(0, imageName.find(".")), scale.first, scale.second);
        }
    }

    cout << "\nResults saved in /output/doc/.\n";
}

int main(int argc, char** argv) {
    //testMain();
    Mat input = imread("../images/Flowers_24bits.bmp");
    if (input.empty()) {
        std::cerr << "Failed to load image.\n";
        return -1;
    }

    double scaleX = 1.5;
    double scaleY = 1.5;

    Mat nn = nearestNeighbor(input, scaleX, scaleY);
    Mat bl = bilinear(input, scaleX, scaleY);
    Mat bc = bicubicCustom(input, scaleX, scaleY, -0.5f);
    Mat lz = lanczos(input, scaleX, scaleY, 3);

    imshow("Original", input);
    imshow("Nearest Neighbor", nn);
    imshow("Bilinear", bl);
    imshow("Bicubic (a=-0.5)", bc);
    imshow("Lanczos (a=3)", lz);

    imwrite("../colorTestOutput/output_nearest.jpg", nn);
    imwrite("../colorTestOutput/output_bilinear.jpg", bl);
    imwrite("../colorTestOutput/output_bicubic.jpg", bc);
    imwrite("../colorTestOutput/output_lanczos.jpg", lz);

    waitKey(0);
    return 0;
}
