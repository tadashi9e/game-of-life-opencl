char get(
    __global unsigned char *map,
    int x,
    int y,
    int width,
    int height) {
  if (x < 0) {
    x += width;
  } else if (x >= width) {
    x -= width;
  }
  if (y < 0) {
    y += height;
  } else if (y >= height) {
    y -= height;
  }
  return map[y * width + x];
}

__kernel void devGolGenerate(
    __global unsigned char *map_in,
    __global unsigned char *map_out,
    __write_only image2d_t image,
    int width,
    int height) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x >= width || y >= height) return;

  int cell_count = 0;

  if (get(map_in, x-1, y-1, width, height)) ++cell_count;
  if (get(map_in, x-1, y  , width, height)) ++cell_count;
  if (get(map_in, x-1, y+1, width, height)) ++cell_count;
  if (get(map_in, x  , y-1, width, height)) ++cell_count;
  if (get(map_in, x  , y+1, width, height)) ++cell_count;
  if (get(map_in, x+1, y-1, width, height)) ++cell_count;
  if (get(map_in, x+1, y  , width, height)) ++cell_count;
  if (get(map_in, x+1, y+1, width, height)) ++cell_count;

  if (get(map_in, x, y, width, height) == 1 &&
      (cell_count > 3 || cell_count < 2)) {
    map_out[y * width + x] = 0;
  } else if (get(map_in, x, y, width, height) == 0 &&
             cell_count == 3) {
    map_out[y * width + x] = 1;
  } else {
    map_out[y * width + x] = get(map_in, x, y, width, height);
  }

  const float r = (map_out[y * width + x] != 0) ? 1.0 : 0.0;
  const float g = ((map_in[y * width + x] != 0) &&
                   (map_out[y * width + x] != 0)) ? 1.0 : 0.0;
  const float b = (map_in[y * width + x] != 0) ? 1.0 : 0.0;
  const float4 pixel = (float4)(r, g, b, 255);
  write_imagef(image, (int2)(x,y), pixel);
}
