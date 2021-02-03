############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################
cmake_minimum_required(VERSION 3.8)

find_package(CUDA QUIET REQUIRED)
find_package(OpenCV REQUIRED)

enable_language(CUDA)
# Necessary to use add_executable instead of deprecated CUDA_ADD_EXECUTABLE

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# collect source files

file( GLOB hdr *.hpp *.h )
file( GLOB cu *.cu )

SET (HW6_files main.cpp loadSaveImage.cpp reference_calc.cpp compare.cpp)

add_executable(HW6 ${HW6_files} ${hdr} ${cu})

# The 4 lines below implement support of cooperative groups in CMake
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ;-rdc=true;-gencode=arch=compute_60,code=sm_60)
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=sm_60")
set_target_properties(HW6 PROPERTIES POSITION_INDEPENDENT_CODE  ON)
set_target_properties(HW6 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)    

target_link_libraries(HW6 ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})

