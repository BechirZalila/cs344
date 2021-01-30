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
#include "timer.h"

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

void denseHisto (unsigned int* const d_vals, //INPUT
		 unsigned int* const d_histo,      //OUTPUT
		 const unsigned int numBins,
		 const unsigned int numElems)
{
  thrust::device_ptr<unsigned int> vals (d_vals);
  thrust::device_ptr<unsigned int> histo (d_histo);
  thrust::counting_iterator<unsigned int> search_begin (0);

  GpuTimer t1;
  t1.Start();
  thrust::sort (vals, vals + numElems);
  thrust::upper_bound (vals, vals + numElems,
		       search_begin, search_begin + numBins,
		       histo);
  thrust::adjacent_difference (histo, histo + numBins, histo);
  t1.Stop();
  printf("Dense Histo ran in: %f msecs.\n", t1.Elapsed());

  //  printVector ("Dense  Histo : ",
  //	       thrust::device_pointer_cast(d_histo),
  //	       thrust::device_pointer_cast(d_histo) + numBins);
}

void sparseHisto (unsigned int* const d_vals,       //INPUT
		  unsigned int* const d_histo,      //OUTPUT
		  const unsigned int numBins,
		  const unsigned int numElems)
{
  thrust::device_ptr<unsigned int> vals (d_vals);
  thrust::device_vector<unsigned int> histo_vals (numBins);
  thrust::device_vector<unsigned int> histo_counts (numBins);

  GpuTimer t2;
  t2.Start();
  thrust::sort (vals, vals + numElems);
  thrust::reduce_by_key (vals, vals + numElems,
			 thrust::constant_iterator<unsigned int>(1),
			 histo_vals.begin(), histo_counts.begin());
  

  thrust::scatter (histo_counts.begin(), histo_counts.end(), histo_vals.begin(),
		   thrust::device_pointer_cast(d_histo));

  // FIXME: For some mysterious reason, the scatter operation does not
  // update the first element of the histogram corresponding to value
  // O.
  thrust::copy (histo_counts.begin(),
		histo_counts.begin()+1,
		thrust::device_pointer_cast(d_histo));
  t2.Stop();
  printf("Sparse Histo ran in: %f msecs.\n", t2.Elapsed());

  //printVector ("Sparse Histo : ", thrust::device_pointer_cast(d_histo),
  //	       thrust::device_pointer_cast(d_histo) + numBins);
  //  printVector ("Histo Vals   : ", histo_vals.begin(), histo_vals.end());
  //printVector ("Histo Counts : ", histo_counts.begin(), histo_counts.end());
}

void computeHistogram(unsigned int* const d_vals, //INPUT
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
  GpuTimer t3;
  switch (method)
    {
    case 0:
      // Launch the simple naive histo
      t3.Start();
      yourHisto<<<blocks, threads>>>(d_vals, d_histo, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      t3.Stop();
      printf("Cuda Histo ran in: %f msecs.\n", t3.Elapsed());
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
