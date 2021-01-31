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

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/counting_iterator.h>

#include "utils.h"

template<typename InputIterator>
void printVector(const char * const msg,
		 InputIterator begin,
		 InputIterator end)
{
  std::cout << msg << "  ";
  thrust::copy(begin, end,
	       std::ostream_iterator<unsigned int>(std::cout, " "));
  std::cout << std::endl;
}

// Very Simple Histo
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

// Slighly better histo
__global__
void betterHisto(const unsigned int* const vals, //INPUT
		 unsigned int* const histo,      //OUPUT
		 int numVals,
		 int numBins)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code

  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  extern __shared__ unsigned int localHisto[]; // Allocated on kernel invocation

  if (myId >= numVals)
    return;

  // Reset local histo.
  if (myId < numBins) {
    localHisto[myId] = 0;
  }
  __syncthreads();

  // Compute local histogram
  unsigned int stride = blockDim.x * gridDim.x;
  int i = myId;
  while (i < numVals) {
    atomicAdd (&localHisto[vals[i]], 1);
    i += stride;
  }
  __syncthreads();

  // Merge histograms. Same mechanism as above:
  if (myId < numBins) {
    atomicAdd (&(histo[myId]), localHisto[myId]);
  }
}

void denseHisto (thrust::device_ptr<unsigned int> &d_vals,
		 thrust::device_ptr<unsigned int> &d_histo,
		 const unsigned int numBins,
		 const unsigned int numElems)
{
  thrust::counting_iterator<unsigned int> search_begin (0);
  thrust::sort (d_vals, d_vals + numElems);
  thrust::upper_bound (d_vals, d_vals + numElems,
		       search_begin, search_begin + numBins,
		       d_histo);
  thrust::adjacent_difference (d_histo, d_histo + numBins, d_histo);

  //  printVector ("Dense  Histo : ",
  //	       thrust::device_pointer_cast(d_histo),
  //	       thrust::device_pointer_cast(d_histo) + numBins);
}

void sparseHisto (thrust::device_ptr<unsigned int> &d_vals,
		  thrust::device_ptr<unsigned int> &d_histo,
		  thrust::device_vector<unsigned int> &d_histo_vals,
		  thrust::device_vector<unsigned int> &d_histo_counts,
		  const unsigned int numBins,
		  const unsigned int numElems)
{
  thrust::sort (d_vals, d_vals + numElems);
  thrust::reduce_by_key (d_vals, d_vals + numElems,
			 thrust::constant_iterator<unsigned int>(1),
			 d_histo_vals.begin(), d_histo_counts.begin());
  

  thrust::scatter (d_histo_counts.begin(), d_histo_counts.end(),
		   d_histo_vals.begin(),
		   thrust::device_pointer_cast(d_histo));

  // FIXME: For some mysterious reason, the scatter operation does not
  // update the first element of the histogram corresponding to value
  // O.

  /*unsigned int a;
  thrust::copy (d_histo_vals.begin(), d_histo_vals.begin() + 1, &a);
  if (a == 0) {
    thrust::copy (d_histo_counts.begin(),
		  d_histo_counts.begin()+1,
		  thrust::device_pointer_cast(d_histo));
		  }*/

  thrust::device_vector<unsigned int> xxxx (numBins);
  thrust::scatter (d_histo_counts.begin(), d_histo_counts.end(),
		   d_histo_vals.begin(), xxxx.begin());

  printVector ("xxxx  Histo : ", xxxx.begin(), xxxx.end());
  printVector ("Dense Histo : ", thrust::device_pointer_cast(d_histo),
	       thrust::device_pointer_cast(d_histo) + numBins);
  //printVector ("Histo Vals   : ", d_histo_vals.begin(), d_histo_vals.end());
  //printVector ("Histo Counts : ", d_histo_counts.begin(), d_histo_counts.end());
}

// Max number of threads per block
const int maxThreadsPerBlock = 1024;

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel

  //if you want to use/launch more than one kernel,
  //feel free

  int threads = maxThreadsPerBlock;
  int blocks = (numElems / maxThreadsPerBlock) + 1;

  yourHisto<<<blocks, threads>>>(d_vals, d_histo, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void computeBetterHistogram(const unsigned int* const d_vals, //INPUT
			    unsigned int* const d_histo,      //OUTPUT
			    const unsigned int numBins,
			    const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel

  //if you want to use/launch more than one kernel,
  //feel free

  int threads = maxThreadsPerBlock;
  int blocks = (numElems / maxThreadsPerBlock) + 1;

  betterHisto<<<blocks, threads, numBins * sizeof(unsigned int)>>>
    (d_vals, d_histo, numElems, numBins);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
