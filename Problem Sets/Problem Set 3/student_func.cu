/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

__global__ void shmem_min_reduce(float * d_out,
				 const float * d_in,
				 const size_t size)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId  = threadIdx.x + blockDim.x * blockIdx.x;
    int tid   = threadIdx.x;

    // Make sure there is no overflow
    if (myId >= size){
      return;
    }

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2, old_s = blockDim.x; s > 0; s >>= 1)
    {
        if (tid < s)
        {
	  sdata[tid] = min (sdata[tid], sdata[tid + s]);
        }

	// If the previous chunk is of an odd size, the last element
	// of the chunk would not be used. The statement below
	// prevents that from happening.
	if ((tid == s - 1) && (old_s % 2 != 0)) {
	  sdata [tid] = max (sdata[tid], sdata [tid + s + 1]);
	}

	old_s = s;
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

__global__ void shmem_max_reduce(float * d_out,
				 const float * d_in,
				 const size_t size)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId  = threadIdx.x + blockDim.x * blockIdx.x;
    int tid   = threadIdx.x;

    // Make sure there is no overflow
    if (myId >= size){
      return;
    }

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2, old_s = blockDim.x; s > 0; s >>= 1)
    {
        if (tid < s)
        {
	  sdata[tid] = max (sdata[tid], sdata[tid + s]);
        }

	// If the previous chunk is of an odd size, the last element
	// of the chunk would not be used. The statement below
	// prevents that from happening.
	if ((tid == s - 1) && (old_s % 2 != 0)) {
	  sdata [tid] = max (sdata[tid], sdata [tid + s + 1]);
	}

	old_s = s;
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

__global__ void simple_histo(unsigned int *d_bins,
			     const float *d_logLuminance,
			     const float min_logLum,
			     const float range_logLum,
			     const size_t size,
			     const size_t numBins)
{
  int myId  = threadIdx.x + blockDim.x * blockIdx.x;
    
  // Make sure there is no overflow
  if (myId >= size){
    return;
  }

  float myItem = d_logLuminance [myId];
  int myBin = (myItem - min_logLum) / range_logLum * numBins;
  atomicAdd(&(d_bins[myBin]), 1);
}

__global__ void stupid_scan (unsigned int *d_cdf, unsigned int *d_histo, int n)
{
  d_cdf[0] = 0;
  for (size_t i = 1; i < n; ++i) {
    d_cdf[i] = d_cdf[i - 1] + d_histo[i - 1];
  }
}

__global__ void naive_scan(unsigned int *g_odata, unsigned int *g_idata, int n)
{
  // Hillis and Steel algo. This is an inclusive scan. To get an
  // exclusive scan, we shift all the elements to the right first.
  
  // From: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
  
  extern __shared__ float temp[]; // allocated on invocation
  int thid = threadIdx.x;
  int pout = 0, pin = 1;
  
  // Load input into shared memory.
  // This is exclusive scan, so shift right by one
  // and set first element to 0
  temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;
  __syncthreads();

  for (int offset = 1; offset < n; offset *= 2) {
      pout = 1 - pout; // swap double buffer indices
      pin = 1 - pout;
      if (thid >= offset)
	temp[pout*n+thid] += temp[pin*n+thid - offset];
      else
	temp[pout*n+thid] = temp[pin*n+thid];
      __syncthreads();
  }

  g_odata[thid] = temp[pout*n+thid]; // write output 

}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement */

  const size_t numPixels = numCols * numRows;
  const int maxThreadsPerBlock = 1024;

  //1) find the minimum and maximum value in the input logLuminance channel
  //   store in min_logLum and max_logLum

  int threads = maxThreadsPerBlock;
  int blocks = (numPixels / maxThreadsPerBlock) + 1;

  float *d_intermediate, *d_out;
  checkCudaErrors(cudaMalloc(&d_intermediate, blocks * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_out, sizeof(float)));;

  // Min reduce
  shmem_min_reduce<<<blocks, threads, threads * sizeof(float)>>>
    (d_intermediate, d_logLuminance, numPixels);

  // Now we're down to one block left, so reduce it
  threads = blocks; // launch one thread for each block in prev step
  blocks = 1;

  shmem_min_reduce<<<blocks, threads, threads * sizeof(float)>>>
    (d_out, d_intermediate, threads);
  checkCudaErrors (cudaMemcpy(&min_logLum,
			      d_out,
			      sizeof (float),
			      cudaMemcpyDeviceToHost));

  // Max reduce
  threads = maxThreadsPerBlock;
  blocks = (numPixels / maxThreadsPerBlock) + 1;
  shmem_max_reduce<<<blocks, threads, threads * sizeof(float)>>>
    (d_intermediate, d_logLuminance, numPixels);

  // Now we're down to one block left, so reduce it
  threads = blocks; // launch one thread for each block in prev step
  blocks = 1;

  shmem_max_reduce<<<blocks, threads, threads * sizeof(float)>>>
    (d_out, d_intermediate, threads);
  checkCudaErrors (cudaMemcpy(&max_logLum,
			      d_out,
			      sizeof (float),
			      cudaMemcpyDeviceToHost));
  
  //2) subtract them to find the range

  float range_logLum = max_logLum - min_logLum;
  
  //3) generate a histogram of all the values in the logLuminance channel using
  //   the formula: bin = (lum[i] - lumMin) / lumRange * numBins

  unsigned int *d_histo;
  
  checkCudaErrors(cudaMalloc(&d_histo, numBins * sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));

  threads = maxThreadsPerBlock;
  blocks = (numPixels / maxThreadsPerBlock) + 1;

  simple_histo<<<blocks, threads, threads * sizeof(float)>>>
    (d_histo, d_logLuminance, min_logLum, range_logLum,  numPixels, numBins);
    
  //4) Perform an exclusive scan (prefix sum) on the histogram to get
  //   the cumulative distribution of luminance values (this should go in the
  //   incoming d_cdf pointer which already has been allocated for you)

  assert (numBins <= maxThreadsPerBlock);

  threads = numBins;
  blocks = 1;
  size_t shmem = 2 * numBins * sizeof (unsigned int);

  
  naive_scan<<<blocks, threads, shmem>>> (d_cdf, d_histo, numBins);
  //stupid_scan<<<1,1>>> (d_cdf, d_histo, numBins);

  const size_t S = numBins;
  unsigned int h_cdf[S];

  checkCudaErrors (cudaMemcpy (h_cdf,
			       d_cdf,
			       numBins * sizeof (unsigned int),
			       cudaMemcpyDeviceToHost));
  for (int k = 0; k < numBins; k++) {
    printf ("%u ", h_cdf [k]);
    if (k % 50 == 0)
      printf ("\n");
  }
  printf ("\n");

  checkCudaErrors(cudaFree(d_intermediate));
  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFree(d_histo));
  

}
