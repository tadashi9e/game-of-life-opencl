all: gol

gol: gol.cpp
	g++ -Wall -I/usr/local/cuda/include gol.cpp -o gol -g -lOpenCL -lglut -lGLEW -lGLU -lGL -fopenmp

clean:
	rm -rf gol
