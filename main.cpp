#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/interpolation.h"
using namespace std;
using namespace cv;


void saveImage(const Mat& img, const String& windowName) {
    imwrite("../output/" + windowName + ".png", img);
    cout << "Saved image: " << windowName << ".png" << endl;
}
Mat getImage(String name) {
    Mat image = imread("../images/"+ name, IMREAD_GRAYSCALE);
    imshow(name, image);
    moveWindow(name, 0, 0);
    return image;
}

int main(int argc, char** argv) {

    // pozele initiale
    Mat scaleUpImg = getImage("star.bmp");
    if (scaleUpImg.empty()) {
        cout << "Could not open or find the image!" << endl;
        return 0;
    }
    Mat scaleDownImg = getImage("binaryflower.png");
    if (scaleDownImg.empty()) {
        cout << "Could not open or find the image!" << endl;
        return 0;
    }

    // Parametrii de scalare
    double scaleUpX = 2.0, scaleUpY = 2.0;     // > 1.0
    double scaleDownX = 0.2, scaleDownY = 0.2; // > 0.0 && < 1.0

    // Nearest neighbor interpolation
    Mat nearest = nearestNeighbor(scaleUpImg, scaleUpX, scaleUpY);
    imshow("Nearest neighbour", nearest);
    moveWindow("Nearest neighbour", 200, 0);

    // Bilinear interpolation
    Mat bil = bilinear(scaleUpImg, scaleUpX, scaleUpY);
    imshow("Bilinear", bil);
    moveWindow("Bilinear", 300, 0);


    // Bicubic interpolation
    Mat bicubic_CatmullRom = bicubicCustom(scaleUpImg, scaleUpX, scaleUpY);  // Catmull-Rom
    Mat bicubic_Sharper = bicubicCustom(scaleUpImg, scaleUpX, scaleUpY, -5.0f);
    Mat bicubic_Smoother = bicubicCustom(scaleUpImg, scaleUpX, scaleUpY, 5.0f);
    imshow("bicubic_CatmullRom", bicubic_CatmullRom);
    moveWindow("bicubic_CatmullRom", 200, 100);
    imshow("bicubic_Sharper", bicubic_Sharper);
    moveWindow("bicubic_Sharper", 400, 100);
    imshow("bicubic_Smoother", bicubic_Smoother);
    moveWindow("bicubic_Smoother", 500, 100);


    // Lanczos interpolation
    Mat lanc = lanczos(scaleDownImg, scaleDownX, scaleDownY);
    Mat lanc_greater = lanczos(scaleDownImg, scaleDownX, scaleDownY,6);
    Mat lanc_lower = lanczos(scaleDownImg, scaleDownX, scaleDownY,1);
    imshow("Lanczos", lanc);
    moveWindow("Lanczos", 300, 100);
    imshow("Lanczos Lower param", lanc_lower);
    moveWindow("Lanczos Lower param", 300, 200);
    imshow("Lanczos Greater param", lanc_greater);
    moveWindow("Lanczos Greater param", 300, 300);



    saveImage(nearest, "Nearest_neighbour");
    saveImage(bil, "Bilinear");
    saveImage(bicubic_CatmullRom, "bicubic_CatmullRom");
    saveImage(bicubic_Sharper, "bicubic_Sharper");
    saveImage(bicubic_Smoother, "bicubic_Smoother");
    saveImage(lanc, "Lanczos");
    saveImage(lanc, "Lanczos Greater param");
    saveImage(lanc, "Lanczos Lower param");
    waitKey();

    return 0;
}
