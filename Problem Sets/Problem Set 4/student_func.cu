//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ void histo_kernel(unsigned int * d_out,
			     unsigned int* const d_in,
			     unsigned int shift,
			     const unsigned int numElems)
{
  // Only 2 bins (one for zeroes and on for ones)
  
  unsigned int mask = 1 << shift;
  
  int myId = threadIdx.x + blockDim.x * blockIdx.x;

  if (myId >= numElems)
    return;

  int bin = (d_in[myId] & mask) >> shift;
  atomicAdd(&d_out[bin], 1);
}

// Blelloch Scan - described in lecture
__global__ void sumscan_kernel(unsigned int * d_in,
			       const size_t numBins,
			       const unsigned int numElems)
{
  int myId = threadIdx.x;
  
  if (myId >= numElems)
    return;

  extern __shared__ float sdata[]; // Allocated on kernel call
  
  sdata[myId] = d_in[myId];
  __syncthreads();            // make sure entire block is loaded!

  for (int d = 1; d < numBins; d *= 2) {
    if (myId >= d) {
      sdata[myId] += sdata[myId - d];
    }
    __syncthreads();
  }
  
  if (myId == 0)
    d_in[0] = 0;
  else
    d_in[myId] = sdata[myId - 1]; //inclusive->exclusive
}

__global__ void makescan_kernel(unsigned int * d_in,
				unsigned int *d_scan,
				unsigned int shift,
				const unsigned int numElems)
{
  unsigned int mask = 1 << shift;
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  
  if (myId >= numElems)
    return;

  d_scan[myId] = ((d_in[myId] & mask) >> shift) ? 0 : 1;
}

__global__ void move_kernel(unsigned int* const d_inputVals,
			    unsigned int* const d_inputPos,
			    unsigned int* const d_outputVals,
			    unsigned int* const d_outputPos,
			    const unsigned int numElems,
			    unsigned int* const d_histogram,
			    unsigned int* const d_scaned,
			    unsigned int shift)
{
  unsigned int mask = 1 << shift;
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  
  if (myId >= numElems)
    return;

  // Algorithm described in 7.4 of http://wykvictor.github.io/2016/04/03/Cuda-2.html 

  int des_id = 0;

  if ((d_inputVals[myId] & mask) >> shift) {
    des_id = myId + d_histogram[1] - d_scaned[myId];
  } else {
    des_id = d_scaned[myId];
  }
  
  d_outputVals[des_id] = d_inputVals[myId];
  d_outputPos[des_id] = d_inputPos[myId];
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  
}
