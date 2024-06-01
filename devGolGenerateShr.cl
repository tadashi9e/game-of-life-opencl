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

bool next_alive(bool old_alive,
                int count,
                int count_weak) {
  if (old_alive) {
    if (count == 2 || count == 3) {
      return true;
    }
    if (count == 1 && count_weak >= 2) {
      return true;
    }
  } else {
    if (count == 3) {
      return true;
    }
    if (count == 2 && count_weak >= 2) {
      return true;
    }
    if (count == 1 && count_weak >= 3) {
      return true;
    }
  }
  return false;
}

#define R 0x01
#define S 0x02
#define P 0x04

__kernel void devGolGenerate(
    __global unsigned char *map_in,
    __global unsigned char *map_out,
    __write_only image2d_t image,
    int width,
    int height) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x >= width || y >= height) return;
  char c;
  int cell_count_r = 0;
  int cell_count_s = 0;
  int cell_count_p = 0;

  c = get(map_in, x-1, y-1, width, height);
  if ((c & R) != 0) ++cell_count_r;
  if ((c & S) != 0) ++cell_count_s;
  if ((c & P) != 0) ++cell_count_p;
  c = get(map_in, x-1, y  , width, height);
  if ((c & R) != 0) ++cell_count_r;
  if ((c & S) != 0) ++cell_count_s;
  if ((c & P) != 0) ++cell_count_p;
  c = get(map_in, x-1, y+1, width, height);
  if ((c & R) != 0) ++cell_count_r;
  if ((c & S) != 0) ++cell_count_s;
  if ((c & P) != 0) ++cell_count_p;
  c = get(map_in, x  , y-1, width, height);
  if ((c & R) != 0) ++cell_count_r;
  if ((c & S) != 0) ++cell_count_s;
  if ((c & P) != 0) ++cell_count_p;
  c = get(map_in, x  , y+1, width, height);
  if ((c & R) != 0) ++cell_count_r;
  if ((c & S) != 0) ++cell_count_s;
  if ((c & P) != 0) ++cell_count_p;
  c = get(map_in, x+1, y-1, width, height);
  if ((c & R) != 0) ++cell_count_r;
  if ((c & S) != 0) ++cell_count_s;
  if ((c & P) != 0) ++cell_count_p;
  c = get(map_in, x+1, y  , width, height);
  if ((c & R) != 0) ++cell_count_r;
  if ((c & S) != 0) ++cell_count_s;
  if ((c & P) != 0) ++cell_count_p;
  c = get(map_in, x+1, y+1, width, height);
  if ((c & R) != 0) ++cell_count_r;
  if ((c & S) != 0) ++cell_count_s;
  if ((c & P) != 0) ++cell_count_p;

  c = get(map_in, x, y, width, height);
  char c2 = 0;
  int cell_count;
  // R
  if (next_alive(c & R, cell_count_r, cell_count_s)) {
    c2 |= R;
  }
  // S
  if (next_alive(c & S, cell_count_s, cell_count_p)) {
    c2 |= S;
  }
  // P
  if (next_alive(c & P, cell_count_p, cell_count_r)) {
    c2 |= P;
  }
  // R > S > P > R
  map_out[y * width + x] =
    (c2 == (R | S | P)) ? 0 :
    (c2 == (R | S)) ? R :
    (c2 == (S | P)) ? S :
    (c2 == (P | R)) ? P :
    c2;
  c = map_out[y * width + x];
  const float r = ((c & R) != 0) ? 1.0 : 0.0;
  const float g = ((c & S) != 0) ? 1.0 : 0.0;
  const float b = ((c & P) != 0) ? 1.0 : 0.0;
  const float4 pixel = (float4)(r, g, b, 255);
  write_imagef(image, (int2)(x,y), pixel);
}
