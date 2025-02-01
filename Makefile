CXX_FLAGS=-Wall -D__CL_ENABLE_EXCEPTIONS -DCL_HPP_TARGET_OPENCL_VERSION=220

all: gol

gol: gol.cpp
	g++ $(CXX_FLAGS) gol.cpp -o gol -g -lglut -lGLEW -lGLU -lGL `pkg-config --libs --cflags OpenCL`

clean:
	rm -rf gol
