############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################
cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET REQUIRED)

enable_language(CUDA)
# Necessary to use add_executable instead of deprecated CUDA_ADD_EXECUTABLE

include_directories(${CUDA_INCLUDE_DIRS})

# collect source files

file(GLOB cu example_block_scan_cum.cu)

add_executable(example_block_scan_cum ${cu})

