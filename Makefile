CXXFLAGS=-g -Wall `pkg-config --cflags OpenCL glut glu gl`
LDFLAGS=`pkg-config --libs OpenCL glut glu gl`

all: gol

gol: gol.cpp
	g++ $(CXXFLAGS) gol.cpp -o gol $(LDFLAGS)

clean:
	rm -rf gol
