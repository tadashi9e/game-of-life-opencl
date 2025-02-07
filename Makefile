CXXFLAGS=-g -Wall
LDFLAGS=`pkg-config --cflags --libs OpenCL glut glu gl`

all: gol

gol: gol.cpp
	g++ $(CXXFLAGS) gol.cpp -o gol $(LDFLAGS)

clean:
	rm -rf gol
