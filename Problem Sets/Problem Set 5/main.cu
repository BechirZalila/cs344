#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <fstream>
#include "utils.h"
#include "timer.h"
#include <cstdio>

#if defined(_WIN16) || defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#else
#include <sys/time.h>
#endif

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/counting_iterator.h>

#include "reference_calc.h"

void computeHistogram(unsigned int *const d_vals,
                      unsigned int* const d_histo,
                      const unsigned int numBins,
                      const unsigned int numElems);

void denseHisto (thrust::device_ptr<unsigned int> &d_vals,
		 thrust::device_ptr<unsigned int> &d_histo,
		 const unsigned int numBins,
		 const unsigned int numElems);

void sparseHisto (thrust::device_ptr<unsigned int> &d_vals,
		  thrust::device_ptr<unsigned int> &d_histo,
		  thrust::device_vector<unsigned int> &d_histo_vals,
		  thrust::device_vector<unsigned int> &d_histo_counts,
		  const unsigned int numBins,
		  const unsigned int numElems);

int main(int argc, char** argv)
{
  const unsigned int numBins = 1024;
  const unsigned int numElems = 10000 * numBins;
  const float stddev = 100.f;

  unsigned int *vals = new unsigned int[numElems];
  unsigned int *h_vals = new unsigned int[numElems];
  unsigned int *h_studentHisto = new unsigned int[numBins];
  unsigned int *h_refHisto = new unsigned int[numBins];

  if (argc != 2) {
    std::cerr << "Usage: ./HW5 method" << std::endl;
    exit (1);
  }

  int method = atoi (argv[1]);

#if defined(_WIN16) || defined(_WIN32) || defined(_WIN64)
  srand(GetTickCount());
#else
  timeval tv;
  gettimeofday(&tv, NULL);

  srand(tv.tv_sec / 60); // To get same generation each minut
#endif

  //make the mean unpredictable, but close enough to the middle
  //so that timings are unaffected
  unsigned int mean = rand() % 100 + 462;

  //Output mean so that grading can happen with the same inputs
  std::cout << mean << std::endl;

  thrust::minstd_rand rng;

  thrust::random::normal_distribution<float> normalDist((float)mean, stddev);

  // Generate the random values
  for (size_t i = 0; i < numElems; ++i) {
    vals[i] = std::min((unsigned int) std::max((int)normalDist(rng), 0), numBins - 1);
  }

  unsigned int *d_vals, *d_histo;

  GpuTimer timer;

  checkCudaErrors(cudaMalloc(&d_vals,    sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&d_histo,   sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));

  checkCudaErrors(cudaMemcpy(d_vals, vals, sizeof(unsigned int) * numElems, cudaMemcpyHostToDevice));

  // For thrust dense and sparse histo implementation
  thrust::device_ptr<unsigned int> d_thrust_vals (d_vals);
  thrust::device_ptr<unsigned int> d_thrust_histo (d_histo);
  thrust::device_vector<unsigned int> d_thrust_histo_vals (numBins);
  thrust::device_vector<unsigned int> d_thrust_histo_counts (numBins);

  switch (method)
    {
    case 0:
      timer.Start();
      computeHistogram(d_vals, d_histo, numBins, numElems);
      timer.Stop();
      break;
    case 1:
      timer.Start();
      denseHisto (d_thrust_vals, d_thrust_histo, numBins, numElems);
      timer.Stop();
      break;
    case 2:
      timer.Start();
      sparseHisto (d_thrust_vals, d_thrust_histo, d_thrust_histo_vals,
		   d_thrust_histo_counts, numBins, numElems);
      timer.Stop();
      break;
    default:
      std::cerr << "   Invalid method: " << method << "." << std::endl;
      exit (1);
      break;
    }
  int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!"
	      << std::endl;
    exit(1);
  }

  // copy the student-computed histogram back to the host
  checkCudaErrors(cudaMemcpy(h_studentHisto, d_histo, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

  //generate reference for the given mean
  timer.Start();
  reference_calculation(vals, h_refHisto, numBins, numElems);
  timer.Stop();
  printf("Ref. code ran in: %f msecs.\n", timer.Elapsed());

  //Now do the comparison
  checkResultsExact(h_refHisto, h_studentHisto, numBins);

  delete[] h_vals;
  delete[] h_refHisto;
  delete[] h_studentHisto;

  cudaFree(d_vals);
  cudaFree(d_histo);

  return 0;
}
