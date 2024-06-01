# game-of-life-opencl (Rock-Paper-Scissors)
A OpenCL implementation of Conway's Game of Life, Rock-Paper-Scissors variant.

## Requirement

* OpenCL >= 2.2
* OpenGL >= 4.4
* X11
* g++ >= 11.4

## Platform
* Linux

## Building
```
make
```

## Execution

```
./gol [-w width] [-h height] [-i interval_millis] [-r Life105_file] [-s Life105_file] [-p Life105_file] [-P]

 -w : field width.
 -h : field height.
 -i : step interval in milli seconds.
 -r : Life1.05 format file ('Rock').
 -s : Life1.05 format file ('Scissors').
 -p : Life1.05 format file ('Paper').
 -P : Paused at start. Will be released by 'p' key.
```

## Example

```
./gol -h 1024 -w 1024 -i 10 -r pattern/glider_gun.life -s pattern/puffer_train.life -p pattern/acorn.life
```

<https://youtu.be/2De6dKQT-TE?si=IuZSVgskHLxjf2ch>
