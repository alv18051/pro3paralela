#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <utility>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

class PGMImage {
public:
    unsigned char* pixels;
    int x_dim, y_dim, num_colors;

    PGMImage(char* fname) {
        x_dim = y_dim = num_colors = 0;
        pixels = NULL;

        FILE* ifile;
        ifile = fopen(fname, "rb");
        if (!ifile) return;

        char* buff = NULL;
        size_t temp;

        fscanf(ifile, "%*s %i %i %i", &x_dim, &y_dim, &num_colors);

        getline(&buff, &temp, ifile);

        assert(x_dim > 1 && y_dim > 1 && num_colors > 1);
        pixels = new unsigned char[x_dim * y_dim];
        fread((void*)pixels, 1, x_dim * y_dim, ifile);

        fclose(ifile);
    }

    PGMImage(int x = 100, int y = 100, int col = 16) {
        num_colors = (col > 1) ? col : 16;
        x_dim = (x > 1) ? x : 100;
        y_dim = (y > 1) ? y : 100;
        pixels = new unsigned char[x_dim * y_dim];
        memset(pixels, 0, x_dim * y_dim);
        assert(pixels);
    }

    ~PGMImage() {
        if (pixels != NULL)
            delete[] pixels;
        pixels = NULL;
    }

    bool write(char* fname) {
        int i, j;
        FILE* ofile;
        ofile = fopen(fname, "w+t");
        if (!ofile) return 0;

        fprintf(ofile, "P5\n%i %i\n%i\n", x_dim, y_dim, num_colors);
        fwrite(pixels, 1, x_dim * y_dim, ofile);
        fclose(ofile);
        return 1;
    }
};
void setPixel(PGMImage &image, int x, int y, unsigned char color) {
    if (x >= 0 && x < image.x_dim && y >= 0 && y < image.y_dim) {
        image.pixels[y * image.x_dim + x] = color;
    }
}


void drawLine(PGMImage &image, int r, float theta, unsigned char color) {
    int x0, y0, x1, y1;
    int width = image.x_dim, height = image.y_dim;

    if (theta < M_PI / 4 || theta > 3 * M_PI / 4) { // More horizontal line
        x0 = 0;
        y0 = r / sin(theta);
        x1 = width - 1;
        y1 = (r - x1 * cos(theta)) / sin(theta);
    } else { // More vertical line
        y0 = 0;
        x0 = r / cos(theta);
        y1 = height - 1;
        x1 = (r - y1 * sin(theta)) / cos(theta);
    }

    // Bresenham's line algorithm to draw the line
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1; 
    int err = dx + dy, e2;

    int maxIterations = width + height; // A safe upper limit for the number of iterations
    int iterations = 0;

    while (true) {
        setPixel(image, x0, y0, color);
        if (x0 == x1 && y0 == y1 || iterations >= maxIterations) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
        iterations++;
    }

}



const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 110;
const float radInc = degreeInc * M_PI / 180;
//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                theta += radInc;
              }
          }
      }
}

//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
//__constant__ float d_Cos[degreeBins];
//__constant__ float d_Sin[degreeBins];

//*****************************************************************
//TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }
//TODO Kernel memoria Constante
// __global__ void GPU_HoughTranConst(...)
// {
//   //TODO
// }

__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
  // Calculate global ID
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;

  if (gloID >= w * h) return; // Check if gloID is within the bounds of the image size

  int xCent = w / 2;
  int yCent = h / 2;
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  if (pic[gloID] > 0)
  {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
      float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
      int rIdx = (r + rMax) / rScale;
      atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
    }
  }
}


int main (int argc, char **argv)
{
  if (argc < 2) {
        printf("Usage: %s <input_image.pgm>\n", argv[0]);
        return -1;
    }
  int i;

  PGMImage inImg(argv[1]);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  float* d_Cos;
  float* d_Sin;

  cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }
  printf ("CPU done\n");

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels;

  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  //1 thread por pixel
  int blockNum = ceil (w * h / 256);
  
  GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

  // get results from device
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel execution time: %f milliseconds\n", milliseconds);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  int threshold = 3500; // Adjust as necessary
  std::vector<std::pair<int, int>> significant_lines;

  for (int r = 0; r < rBins; r++) {
        for (int theta = 0; theta < degreeBins; theta++) {
            if (cpuht[r * degreeBins + theta] > threshold) {
                significant_lines.push_back(std::make_pair(r, theta));
            }
        }
  }
  printf("Number of significant lines: %d\n", significant_lines.size());

printf("Drawing lines...\n");
  unsigned char lineColor = 255; // White color for lines
  for (auto &line : significant_lines) {
    int r = line.first;
    float theta = line.second * radInc; // Convert bin number to angle in radians
    drawLine(inImg, r, theta, lineColor);
  }
  printf("Writing output image...\n");

  //inImg.write("modified_image.pgm");
  cv::Mat imageMat(inImg.y_dim, inImg.x_dim, CV_8UC1, inImg.pixels);
  cv::imwrite("output.png", imageMat);

  // compare CPU and GPU results
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }

  printf("Done!\n");

  free(cpuht);
  free(pcCos);
  free(pcSin);
  free(h_hough); 

  cudaFree(d_Cos);
  cudaFree(d_Sin);
  cudaFree(d_in);
  cudaFree(d_hough);

  return 0;
}