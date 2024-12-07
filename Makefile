CUDA_HEADER_DIR=/usr/local/cuda-10.2/include
CXX_FLAGS=-Wall -D__CL_ENABLE_EXCEPTIONS

all: gol

gol: gol.cpp
	g++ $(CXX_FLAGS) -I$(CUDA_HEADER_DIR) gol.cpp -o gol -g -lOpenCL -lglut -lGLEW -lGLU -lGL -fopenmp

clean:
	rm -rf gol
