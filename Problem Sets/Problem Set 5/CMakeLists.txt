############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################
cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET REQUIRED)
find_package(OpenCV REQUIRED)

enable_language(CUDA)
# Necessary to use add_executable instead of deprecated CUDA_ADD_EXECUTABLE

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# collect source files

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  main.cu student.cu)

SET (HW5_files reference_calc.cpp)

add_executable(HW5 ${HW5_files} ${hdr} ${cu})
target_link_libraries(HW5 ${OpenCV_LIBS})

add_executable(sample_hist histogram.cu)

