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

void denseHisto (const unsigned int* const d_vals, //INPUT
		 unsigned int* const d_histo,      //OUTPUT
		 const unsigned int numBins,
		 const unsigned int numElems)
{
  thrust::device_ptr<unsigned int> vals =
    thrust::device_pointer_cast ((unsigned int *)d_vals);
  thrust::device_vector<unsigned int> sorted_data (numElems);
  thrust::copy (vals, vals + numElems, sorted_data.begin());
  thrust::sort (sorted_data.begin(), sorted_data.end());
  
  thrust::device_vector<unsigned int> histo (numBins);
  thrust::counting_iterator<unsigned int> search_begin (0);
  thrust::upper_bound (sorted_data.begin(), sorted_data.end(),
		       search_begin, search_begin + numBins,
		       histo.begin());
  
  thrust::adjacent_difference (histo.begin(), histo.end(), histo.begin());
  thrust::copy (histo.begin(), histo.end(),
		thrust::device_pointer_cast(d_histo)); 
}

void sparseHisto (const unsigned int* const d_vals, //INPUT
		  unsigned int* const d_histo,      //OUTPUT
		  const unsigned int numBins,
		  const unsigned int numElems)
{

  // FIXME: still not done yet!
  thrust::device_ptr<unsigned int> vals =
    thrust::device_pointer_cast ((unsigned int *)d_vals);
  thrust::device_vector<unsigned int> sorted_data (numElems);
  thrust::copy (vals, vals + numElems, sorted_data.begin());
  thrust::sort (sorted_data.begin(), sorted_data.end());
  
  thrust::device_vector<unsigned int> histo_vals (numBins);
  thrust::device_vector<unsigned int> histo_counts (numBins);

  thrust::reduce_by_key (sorted_data.begin(), sorted_data.end(),
			 thrust::constant_iterator<unsigned int>(1),
			 histo_vals.begin(), histo_counts.begin());

  thrust::scatter (histo_counts.begin(), histo_counts.end(),
		   histo_vals.begin(),
		   thrust::device_pointer_cast(d_histo)); 
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
      // Launch the simple naive histo
      yourHisto<<<blocks, threads>>>(d_vals, d_histo, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      break;
    case 1:
      // Dense Histogram using binary search
      denseHisto (d_vals, d_histo, numBins, numElems);
      break;
    case 2:
      // Sparse histogram using reduce_by_key
      sparseHisto (d_vals, d_histo, numBins, numElems);
      break;
    default:
      std::cerr << "   Invalid method: " << method << "." << std::endl;
      exit (1);
      break;
    }
}
