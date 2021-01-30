/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code

  int myId = threadIdx.x + blockDim.x * blockIdx.x;

  if (myId >= numVals)
    return;

  atomicAdd (&(histo[vals[myId]]), 1);
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems,
		      const int method)
{
  //TODO Launch the yourHisto kernel

  //if you want to use/launch more than one kernel,
  //feel free

  const int maxThreadsPerBlock = 1024;

  int threads = maxThreadsPerBlock;
  int blocks = (numElems / maxThreadsPerBlock) + 1;

  switch (method)
    {
    case 0:
      yourHisto<<<blocks, threads>>>(d_vals, d_histo, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      break;
    case 1:
      break;
    case 2:
      break;
    default:
      break;
    }
}
