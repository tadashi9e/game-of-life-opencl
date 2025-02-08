# game-of-life-opencl
A OpenCL implementation of Conway's Game of Life

## Requirement
* OpenCL >= 1.1
* OpenGL >= 4.4
* X11
* gcc >= 4.9

## Platform
* Linux

## Building
```
make
```

## Execution

```
gol [-d device_index] [-w width] [-h height] [-i interval_millis] [-f Life105_file] [-P]
 -d, --device    : Select compute device.
 -w, --width     : Field width.
 -h, --height    : Field height.
 -i, --interval  : Step interval in milli seconds.
 -f, --file      : Life1.05 format file.
 -P, --pause     : Pause at start. Will be released by 'p' key.
```
