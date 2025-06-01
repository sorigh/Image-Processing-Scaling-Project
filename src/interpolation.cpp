#include "interpolation.h"
#include <cmath>

// Clamp helper (pt accesarea pixelilori out of bound)
// ne asiguram ca nu accesam zone invalide
inline uchar getPixel(const Mat& img, int x, int y) {
    // valoarea pixelului sau 0 in caz ca nu e o valoare valida
    x = std::max(0, std::min(x, img.cols - 1));
    y = std::max(0, std::min(y, img.rows - 1));
    return img.at<uchar>(y, x);
}

// Adapted my clamp helper to return a vector of all color values in that pixel
inline Vec3b getColorPixel(const Mat& img, int x, int y) {
    // valoarea pixelului sau 0 in caz ca nu e o valoare valida
    x = std::max(0, std::min(x, img.cols - 1));
    y = std::max(0, std::min(y, img.rows - 1));
    return img.at<Vec3b>(y, x);
}

/**
 * Interpolare Nearest Neighbor
 *
 * Cea mai simpla si rapida metoda de scalare a unei imagini.
 * Pentru fiecare pixel din imaginea output, se selecteaza pixelul cel mai apropiat
 * din imaginea originala.
 *
 * Avantaje:
 * - computare foarte rapida (doar rotunjire).
 * - nu modifica valorile originale ale pixelilor, util pentru imagini binare, coduri de bare etc.
 *
 * Dezavantaje:
 * - poate duce la efecte vizuale pixelate, mai ales la mariri semnificative.
 *
 * @param input Imaginea de intrare
 * @param scaleX Factor de scalare pe axa X
 * @param scaleY Factor de scalare pe axa Y
 * @return Imaginea scalata folosind metoda nearest-neighbor
 */

Mat nearestNeighbor(const Mat& input, double scaleX, double scaleY) {
    // dimensiunea imaginii output
    int newW = input.cols * scaleX;
    int newH = input.rows * scaleY;
    Mat output(newH, newW, input.type()); // supports color

    //loop pe imaginea output
    for (int y = 0; y < newH; ++y) {
        for (int x = 0; x < newW; ++x) {
            int srcX = std::min(static_cast<int>(x / scaleX), input.cols - 1);
            int srcY = std::min(static_cast<int>(y / scaleY), input.rows - 1);

            //static_cast<int> trunchiaza rezultatul la cel mai apropiat
            // pixel (sus-stanga)
            //x / scaleX asociere intre pixel original si pixel nou
            if (input.channels() == 1) {
                output.at<uchar>(y, x) = input.at<uchar>(srcY, srcX);
            } else {
                output.at<Vec3b>(y, x) = input.at<Vec3b>(srcY, srcX);
            }
        }
    }
    return output;
}

/**
 * Interpolare Biliniara
 *
 * Foloseste media ponderata a celor 4 pixeli vecini pentru a estima val noului pixel.
 * Este o metoda intermediara intre nearest-neighbor si bicubic.
 *
 * Avantaje:
 * - rezultate mai putin pixelate decat nearest-neighbor (tip gradient).
 * - papida si usor de implementat.
 *
 * Dezavantaje:
 * - Poate introduce efecte de blur (margini clare estompate).
 *
 * @param input Imaginea de intrare
 * @param scaleX Factor de scalare pe axa X
 * @param scaleY Factor de scalare pe axa Y
 * @return Imaginea scalata folosind interpolare biliniara
 */
Mat bilinear(const Mat& input, double scaleX, double scaleY) {
    // dimensiunea imaginii output
    int newW = input.cols * scaleX;
    int newH = input.rows * scaleY;
    Mat output(newH, newW, input.type()); // can support color

    //loop pe imaginea output
    for (int y = 0; y < newH; ++y) {
        for (int x = 0; x < newW; ++x) {
            // calc coordonatele din imaginea input
            float gx = x / scaleX;
            float gy = y / scaleY;

            // floor = cea mai mare val intreaga
            int x0 = floor(gx);
            int y0 = floor(gy);
            // vecin right
            int x1 = x0 + 1;
            // vecin bottom
            int y1 = y0 + 1;
            //diferenta fractionala pt interpolare
            float dx = gx - x0; //diferenta orizontala
            float dy = gy - y0; //diferenta verticala

            //gray scale remains the same
            if (input.channels() == 1) {
                uchar p00 = getPixel(input, x0, y0);
                uchar p10 = getPixel(input, x1, y0);
                uchar p01 = getPixel(input, x0, y1);
                uchar p11 = getPixel(input, x1, y1);

                float val = (1 - dx) * (1 - dy) * p00 + dx * (1 - dy) * p10 +
                            (1 - dx) * dy * p01 + dx * dy * p11;
                output.at<uchar>(y, x) = static_cast<uchar>(val);
            } else {
                // color image -> more color chanels
                Vec3b res;
                // for each chanel
                for (int c = 0; c < 3; c++) {
                    // 4 surrounding pixels
                    uchar p00 = getColorPixel(input, x0, y0)[c]; // top-left
                    uchar p10 = getColorPixel(input, x1, y0)[c]; // top-right
                    uchar p01 = getColorPixel(input, x0, y1)[c]; // bottom-left
                    uchar p11 = getColorPixel(input, x1, y1)[c]; // bottom-right

                    //interpolare pe baza distantei (formula)
                    float value =
                        (1 - dx) * (1 - dy) * p00 +
                        dx * (1 - dy) * p10 +
                        (1 - dx) * dy * p01 +
                        dx * dy * p11;
                    res[c] = saturate_cast<uchar>(value);
                }
                output.at<Vec3b>(y, x) = res;
            }
        }
    }
    return output;
}

// o functie folosita pentru a calcula ponderile aplicate
// pe fiecare pixel vecin. val ponderata a unui pixel in functie
// de distanta lui fata de pixelul original.
// bicubic convolution algorithm
// sources for the kernel:
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
// https://pixinsight.com/doc/docs/InterpolationAlgorithms/InterpolationAlgorithms.html
// parametrizare: a in general -0.5 sau -0.75.
float cubicKernel(float x, float a) {
    x = abs(x); // non negative vals
    if (x <= 1)
        // distanta mica
        return (a + 2) * x * x * x - (a + 3) * x * x + 1;
    else if (x < 2)
        return a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a;
    return 0.0f; //distanta mare, nu influenteaza pixelul
}

/**
 * Interpolare Bicubica (cu kernel personalizabil)
 *
 * Foloseste 16 pixeli vecini (4x4) si un kernel cubic definit de parametru a.
 * Cand a = -0.5, se obtine spline-ul Catmull-Rom, o alegere comuna.
 *
 * Functia cubicKernel() defineste forma kernelui:
 * - a = -0.5 → Catmull-Rom spline
 * - a < -0.5 → mai neted (blur)
 * - a > -0.5 → mai accentuat (sharpening)
 *
 * Avantaje:
 * - calitate superioara imaginii rezultate.
 * - margini si detalii mai bine conservate fata de biliniar.
 *
 * Dezavantaje:
 * - Mai consumator de resurse (si mai lent in executie).
 *
 * @param input Imaginea de intrare.
 * @param scaleX Factor de scalare pe axa X.
 * @param scaleY Factor de scalare pe axa Y.
 * @param a Parametrul nucleului bicubic (default: -0.5)
 * @return Imaginea scalata folosind interpolare bicubica.
 */
Mat bicubicCustom(const Mat& input, double scaleX, double scaleY, float a) {
    // dimensiunea imaginii output
    int newW = input.cols * scaleX;
    int newH = input.rows * scaleY;
    Mat output(newH, newW, input.type()); // can support color

    for (int y = 0; y < newH; ++y) {
        for (int x = 0; x < newW; ++x) {
            // calc coordonatele din imaginea input
            float gx = x / scaleX;
            float gy = y / scaleY;
            // floor = cea mai mare val intreaga
            int x0 = floor(gx);
            int y0 = floor(gy);
            //diferenta fractionala pt interpolare
            float dx = gx - x0; //diferenta orizontala
            float dy = gy - y0; //diferenta verticala

            if (input.channels() == 1) {
                float sum = 0, weightSum = 0;
                for (int m = -1; m <= 2; ++m) {
                    for (int n = -1; n <= 2; ++n) {
                        float w = cubicKernel(dx - m, a) * cubicKernel(dy - n, a);
                        sum += getPixel(input, x0 + m, y0 + n) * w;
                        weightSum += w;
                    }
                }
                output.at<uchar>(y, x) = saturate_cast<uchar>(sum / weightSum);
            } else {
                // sum is now a vector
                Vec3f sum = Vec3f(0, 0, 0);
                float weightSum = 0;
                // for each color chanel
                    // 4x4 neighbourhood
                for (int m = -1; m <= 2; ++m) {
                    for (int n = -1; n <= 2; ++n) {
                        //folosim kerneul cubic parametrizat pentru x si y
                        float w = cubicKernel(dx - m, a) * cubicKernel(dy - n, a);
                        Vec3b px = getColorPixel(input, x0 + m, y0 + n);
                        for (int c = 0; c < 3; ++c) {
                            sum[c] += px[c] * w;
                        }
                        weightSum += w; // ptr a impartii la final
                    }
                }
                Vec3b result;
                for (int c = 0; c < 3; ++c) {
                    result[c] = saturate_cast<uchar>(sum[c] / weightSum);
                }
                output.at<Vec3b>(y, x) = result;
                }
            }
        }
    return output;
}
// Lanczos kernel
// https://pixinsight.com/doc/docs/InterpolationAlgorithms/InterpolationAlgorithms.html
// Formula: sinc(x) = sin(πx) / (πx), minimizeaza aliasing artifacts.
float sinc(float x) {
    if (x == 0) return 1.0;
    x *= CV_PI;
    return sin(x) / x;
}

// interpolare mai smooth pe baza functiei sinc
// Formula: Lanczos function: sinc(x) * sinc(x / n), unde in general se considera n = 3.
// cunoscut pentru prezervarea detaliilor fine si reducerea aliasarei
float lanczosKernel(float x, int n = 3) {
    x = fabs(x); // val pozitive
    if (x <= n) return sinc(x) * sinc(x / n); //function
    return 0; //fara contributii de la kernel
}
/**
 * Interpolare Lanczos
 *
 * Metoda de interpolare avansata folosind o functie sinc trunchiata, aplicata pe o vecinatate 6x6 pentru a = 3, adică m = -2 până la +3 (6 valori).
 * Este considerata una dintre cele mai precise metode pentru scalare, in special pentru micsorari.
 *
 * Avantaje:
 * - Foarte precisa, cu pastrarea detaliilor si reducerea aliasing-ului.
 * - Rezultate clare si estetice.
 *
 * Dezavantaje:
 * - Computare lenta din cauza functiilor trigonometrice.
 * - Poate introduce artefacte in unele cazuri (ex: "ringing").
 *
 * @param input Imaginea de intrare (grayscale).
 * @param scaleX Factor de scalare pe axa X.
 * @param scaleY Factor de scalare pe axa Y.
 * @return Imaginea scalata folosind interpolare Lanczos cu a = 3.
 */
Mat lanczos(const Mat& input, double scaleX, double scaleY, int a) {
    // dimensiunea imaginii output
    int newW = input.cols * scaleX;
    int newH = input.rows * scaleY;
    Mat output(newH, newW, input.type()); // can support color

    for (int y = 0; y < newH; ++y) {
        for (int x = 0; x < newW; ++x) {
            // calc coordonatele din imaginea input
            float gx = x / scaleX;
            float gy = y / scaleY;
            // floor = cea mai mare val intreaga
            int x0 = floor(gx);
            int y0 = floor(gy);

            if (input.channels() == 1) {
                float sum = 0, weightSum = 0;
                for (int m = -a + 1; m <= a; ++m) {
                    for (int n = -a + 1; n <= a; ++n) {
                        float dx = gx - (x0 + m);
                        float dy = gy - (y0 + n);
                        float w = lanczosKernel(dx, a) * lanczosKernel(dy, a);
                        sum += getPixel(input, x0 + m, y0 + n) * w;
                        weightSum += w;
                    }
                }
                output.at<uchar>(y, x) = saturate_cast<uchar>(sum / weightSum);
            } else {
                Vec3f sum = Vec3f(0, 0, 0);
                float weightSum = 0;
                for (int m = -a + 1; m <= a; ++m) {
                    for (int n = -a + 1; n <= a; ++n) {
                        float dx = gx - (x0 + m);
                        float dy = gy - (y0 + n);
                        float w = lanczosKernel(dx, a) * lanczosKernel(dy, a);
                        Vec3b px = getColorPixel(input, x0 + m, y0 + n);
                        for (int c = 0; c < 3; ++c) sum[c] += px[c] * w;
                        weightSum += w;
                    }
                }
                Vec3b result;
                for (int c = 0; c < 3; ++c)
                    result[c] = saturate_cast<uchar>(sum[c] / weightSum);
                output.at<Vec3b>(y, x) = result;
            }
        }
    }
    return output;
}
