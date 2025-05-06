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
    Mat output(newH, newW, CV_8UC1); // grayscale

    //loop pe imaginea output
    for (int y = 0; y < newH; ++y) {
        for (int x = 0; x < newW; ++x) {
            int srcX = static_cast<int>(x / scaleX);
            int srcY = static_cast<int>(y / scaleY);
            //static_cast<int> trunchiaza rezultatul la cel mai apropiat
            // pixel (sus-stanga)
            //x / scaleX asociere intre pixel original si pixel nou
            output.at<uchar>(y, x) = input.at<uchar>(srcY, srcX);
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
    Mat output(newH, newW, CV_8UC1); // grayscale

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

            // 4 surrounding pixels
            uchar p00 = getPixel(input, x0, y0); // top-left
            uchar p10 = getPixel(input, x1, y0); // top-right
            uchar p01 = getPixel(input, x0, y1); // bottom-left
            uchar p11 = getPixel(input, x1, y1); // bottom-right

            //interpolare pe baza distantei (formula)
            float value =
                (1 - dx) * (1 - dy) * p00 +
                dx * (1 - dy) * p10 +
                (1 - dx) * dy * p01 +
                dx * dy * p11;

            output.at<uchar>(y, x) = static_cast<uchar>(value);
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
// parametrizare: a in general -0.5 (Keys) sau -0.75.
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
 * - a = -0.5 → Catmull-Rom spline (standard, Keys)
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
    Mat output(newH, newW, CV_8UC1); // grayscale

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

            //init sumele
            // weight = mai mare = mai aproape, influenteaza mai mult pixelul care trebuie aproximat
            float sum = 0;
            float weightSum = 0;
            // 4x4 neighbourhood
            for (int m = -1; m <= 2; ++m) {
                for (int n = -1; n <= 2; ++n) {
                    //folosim kerneul cubic parametrizat pentru x si y
                    float w = cubicKernel(dx - m, a) * cubicKernel(dy - n, a);
                    sum += getPixel(input, x0 + m, y0 + n) * w; //adaugam pixelul in suma
                    weightSum += w; // ptr a impartii la final
                }
            }

            // normalizam dupa weight - ul total si asignam valoarea pixelului output
            output.at<uchar>(y, x) = saturate_cast<uchar>(sum / weightSum);
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
    x = abs(x); // val pozitive
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
    Mat output(newH, newW, CV_8UC1); // grayscale

    for (int y = 0; y < newH; ++y) {
        for (int x = 0; x < newW; ++x) {
            // calc coordonatele din imaginea input
            float gx = x / scaleX;
            float gy = y / scaleY;
            // floor = cea mai mare val intreaga
            int x0 = floor(gx);
            int y0 = floor(gy);


            //init sumele
            // weight = mai mare = mai aproape, influenteaza mai mult pixelul care trebuie aproximat
            float sum = 0;
            float weightSum = 0;

            for (int m = -a + 1; m <= a; ++m) {
                for (int n = -a + 1; n <= a; ++n) {
                    float dx = gx - (x0 + m); // diferenta orizontala
                    float dy = gy - (y0 + n); // diferenta verticala
                    float w = lanczosKernel(dx, a) * lanczosKernel(dy, a);
                    sum += getPixel(input, x0 + m, y0 + n) * w; //cu pondere
                    weightSum += w; // ptr a impartii la final
                }
            }

            output.at<uchar>(y, x) = saturate_cast<uchar>(sum / weightSum);
        }
    }

    return output;
}
