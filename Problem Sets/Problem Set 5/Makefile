NVCC=nvcc
NVCC_OPTS=-O3 -arch=sm_75 -Xcompiler -Wall -Xcompiler -Wextra -m64
GCC_OPTS=-O3 -Wall -Wextra -m64

OPENCV_LIBPATH=${shell pkg-config --libs opencv}
OPENCV_INCLUDEPATH=${shell pkg-config --cflags opencv}

CUDA_INCLUDEPATH=/usr/local/cuda-10.1/include

histo: main.cu reference_calc.o student.o Makefile
	nvcc -o HW5 main.cu reference_calc.o student.o $(NVCC_OPTS)

student.o: student.cu
	nvcc -c student.cu $(NVCC_OPTS)

reference_calc.o: reference_calc.cpp reference_calc.h
	g++ -c reference_calc.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

clean:
	rm -f *.o hw *.bin
