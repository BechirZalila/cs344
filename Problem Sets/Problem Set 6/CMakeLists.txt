############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################
cmake_minimum_required(VERSION 2.8)

#find_package(CUDA QUIET REQUIRED)
enable_language(CUDA)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# collect source files

file( GLOB hdr *.hpp *.h )
file( GLOB cu *.cu )

SET (HW6_files main.cpp loadSaveImage.cpp reference_calc.cpp compare.cpp)

add_executable(HW6 ${HW6_files} ${hdr} ${cu})
target_link_libraries(HW5 ${OpenCV_LIBS})

