kernel void update_d2q9_bgk(__constant int *s, 
                            int even, float omega, __constant float *g,
                            global float *f, global float *rho,
                            global float *vel, global int *cell) {

  const int qs[9][2] = {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {-1, 1}, {1, -1}};
  
  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);

  if (ix >= s[0] || iy >= s[1])
    return;

  const size_t ii = ix * s[1] + iy; // 1d idx for arrays

  const int c = cell[ii];
  const bool wall = (c & 1);
  const bool fixed = (c & 6);

  if (wall) {
    // wall => do nothing
    return;
  }

  int off[9];

#pragma unroll
  for (int i = 0; i < 9; i++) {
    int ix_ = (ix + qs[i][0] + s[0]) % s[0];
    int iy_ = (iy + qs[i][1] + s[1]) % s[1];
    off[i] = (ix_ * s[1] + iy_) * 9;
  }

  float f_[9];
  if (even) {
    f_[0] = f[off[0] + 0];
    f_[1] = f[off[2] + 1];
    f_[2] = f[off[1] + 2];
    f_[3] = f[off[4] + 3];
    f_[4] = f[off[3] + 4];
    f_[5] = f[off[6] + 5];
    f_[6] = f[off[5] + 6];
    f_[7] = f[off[8] + 7];
    f_[8] = f[off[7] + 8];
  } else {
    f_[0] = f[off[0] + 0];
    f_[1] = f[off[0] + 2];
    f_[2] = f[off[0] + 1];
    f_[3] = f[off[0] + 4];
    f_[4] = f[off[0] + 3];
    f_[5] = f[off[0] + 6];
    f_[6] = f[off[0] + 5];
    f_[7] = f[off[0] + 8];
    f_[8] = f[off[0] + 7];
  }

  // calc moments
  float r =
      f_[0] + f_[1] + f_[2] + f_[3] + f_[4] + f_[5] + f_[6] + f_[7] + f_[8];
  float vx = (f_[1] - f_[2] + f_[5] - f_[6] - f_[7] + f_[8]) / r;
  float vy = (f_[3] - f_[4] + f_[5] - f_[6] + f_[7] - f_[8]) / r;

  vx += (g[0] / omega);
  vy += (g[1] / omega);

  if (fixed) {
    omega = 1.0;
    r = rho[ii];
    vx = vel[ii * 2 + 0];
    vy = vel[ii * 2 + 1];
  }

  const float vv = vx * vx + vy * vy;
  const float vxx = vx * vx;
  const float vyy = vy * vy;
  const float vxy = vx * vy;

  // calc equilibrium & collide
  // clang-format off
  f_[0] += omega * (r * (2.0 / 9.0) * (2.0 - 3.0 * vv) - f_[0]);
  f_[1] += omega * (r * (1.0 / 18.0) * (2.0 + 6.0 * vx + 9.0 * vxx - 3.0 * vv) - f_[1]);
  f_[2] += omega * (r * (1.0 / 18.0) * (2.0 - 6.0 * vx + 9.0 * vxx - 3.0 * vv) - f_[2]);
  f_[3] += omega * (r * (1.0 / 18.0) * (2.0 + 6.0 * vy + 9.0 * vyy - 3.0 * vv) - f_[3]);
  f_[4] += omega * (r * (1.0 / 18.0) * (2.0 - 6.0 * vy + 9.0 * vyy - 3.0 * vv) - f_[4]);
  f_[5] += omega * (r * (1.0 / 36.0) * (1.0 + 3.0 * (vx + vy) + 9.0 * vxy + 3.0 * vv) - f_[5]);
  f_[6] += omega * (r * (1.0 / 36.0) * (1.0 - 3.0 * (vx + vy) + 9.0 * vxy + 3.0 * vv) - f_[6]);
  f_[7] += omega * (r * (1.0 / 36.0) * (1.0 + 3.0 * (vy - vx) - 9.0 * vxy + 3.0 * vv) - f_[7]);
  f_[8] += omega * (r * (1.0 / 36.0) * (1.0 - 3.0 * (vy - vx) - 9.0 * vxy + 3.0 * vv) - f_[8]);
  // clang-format on

  // write back to same locations
  if (even) {
    f[off[0] + 0] = f_[0];
    f[off[2] + 1] = f_[2];
    f[off[1] + 2] = f_[1];
    f[off[4] + 3] = f_[4];
    f[off[3] + 4] = f_[3];
    f[off[6] + 5] = f_[6];
    f[off[5] + 6] = f_[5];
    f[off[8] + 7] = f_[8];
    f[off[7] + 8] = f_[7];
  } else {
#pragma unroll
    for (int i = 0; i < 9; i++) {
      f[off[0] + i] = f_[i];
    }
  }

  rho[ii] = r;
  vel[ii * 2 + 0] = vx;
  vel[ii * 2 + 1] = vy;
}

kernel void update_d2q5_bgk(global int *s, int even, float omega,
                            global float *f, global float *val, global float *vel,
                            global int *cell) {
  const int qs[5][2] = {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};

  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  // const size_t sx = get_global_size(0);
  // const size_t sy = get_global_size(1);

  if (ix >= s[0] || iy >= s[1])
    return;

  const size_t ii = ix * s[1] + iy; // 1d idx for arrays

  const size_t iv = ii * 2;  // offset for vel
  const size_t if_ = ii * 5; // offset for f

  const int c = cell[ii];
  const bool wall = (c & 1);
  const bool fixed = (c & 8);

  if (wall) {
    // wall => do nothing
    return;
  }

  int off[5];

#pragma unroll
  for (int i = 0; i < 5; i++) {
    int ix_ = (ix + qs[i][0] + s[0]) % s[0];
    int iy_ = (iy + qs[i][1] + s[1]) % s[1];
    off[i] = (ix_ * s[1] + iy_) * 5;
  }

  float f_[5];
  if (even) {
    f_[0] = f[off[0] + 0];
    f_[1] = f[off[2] + 1];
    f_[2] = f[off[1] + 2];
    f_[3] = f[off[4] + 3];
    f_[4] = f[off[3] + 4];
  } else {
    f_[0] = f[off[0] + 0];
    f_[1] = f[off[0] + 2];
    f_[2] = f[off[0] + 1];
    f_[3] = f[off[0] + 4];
    f_[4] = f[off[0] + 3];
  }

  // calc moments
  float C = f_[0] + f_[1] + f_[2] + f_[3] + f_[4];
  float vx = vel[iv + 0];
  float vy = vel[iv + 1];

  if (fixed) {
    omega = 1.0;
    C = val[ii];
  }

  const float vv = vx * vx + vy * vy;
  const float vxx = vx * vx;
  const float vyy = vy * vy;
  const float vxy = vx * vy;

  // calc equilibrium & collide
  // clang-format off
  f_[0] += omega * (C * (1.0 / 6.0) * (2.0 - 3.0 * vv) - f_[0]);
  f_[1] += omega * (C * (1.0 / 12.0) * (2.0 + 6.0 * vx + 9.0 * vxx - 3.0 * vv) - f_[1]);
  f_[2] += omega * (C * (1.0 / 12.0) * (2.0 - 6.0 * vx + 9.0 * vxx - 3.0 * vv) - f_[2]);
  f_[3] += omega * (C * (1.0 / 12.0) * (2.0 + 6.0 * vy + 9.0 * vyy - 3.0 * vv) - f_[3]);
  f_[4] += omega * (C * (1.0 / 12.0) * (2.0 - 6.0 * vy + 9.0 * vyy - 3.0 * vv) - f_[4]);
  // clang-format on

  // write back to same locations
  if (even) {
    f[off[0] + 0] = f_[0];
    f[off[2] + 1] = f_[2];
    f[off[1] + 2] = f_[1];
    f[off[4] + 3] = f_[4];
    f[off[3] + 4] = f_[3];
  } else {
#pragma unroll
    for (int i = 0; i < 5; i++) {
      f[off[0] + i] = f_[i];
    }
  }

  val[ii] = C;
}

kernel void update_d3q27_bgk(int even, float omega, float gx, float gy,
                             float gz, global float *f, global float *rho,
                             global float *vel, global int *cell,
                             global int *qs, int sx, int sy, int sz) {
  const int ix = get_global_id(0);
  const int iy = get_global_id(1);
  const int iz = get_global_id(2);

  if (ix >= sx || iy >= sy || iz >= sz)
    return;

  const int ii = iz * +iy * sz + ix * sz * sy;

  const int c = cell[ii];
  const int wall = (c & 1);
  const int fixed = (c & 6);

  if (wall) {
    // wall => do nothing
    return;
  }

  int off[27];

#pragma unroll
  for (int i = 0; i < 27; i++) {
    const int ix_ = (ix + qs[27 * i + 0] + sx) % sx;
    const int iy_ = (iy + qs[27 * i + 1] + sy) % sy;
    const int iz_ = (iz + qs[27 * i + 2] + sz) % sz;

    off[i] = (0 * sx + ix_ * sy + iy_ * sz + iz_) * 27;
  }

  float f_[27];
  if (even) {
    f_[0] = f[off[0] + 0];
    f_[1] = f[off[2] + 1];
    f_[2] = f[off[1] + 2];
    f_[3] = f[off[6] + 3];
    f_[4] = f[off[8] + 4];
    f_[5] = f[off[7] + 5];
    f_[6] = f[off[3] + 6];
    f_[7] = f[off[5] + 7];
    f_[8] = f[off[4] + 8];
    f_[9] = f[off[18] + 9];
    f_[10] = f[off[20] + 10];
    f_[11] = f[off[19] + 11];
    f_[12] = f[off[24] + 12];
    f_[13] = f[off[26] + 13];
    f_[14] = f[off[25] + 14];
    f_[15] = f[off[21] + 15];
    f_[16] = f[off[23] + 16];
    f_[17] = f[off[22] + 17];
    f_[18] = f[off[9] + 18];
    f_[19] = f[off[11] + 19];
    f_[20] = f[off[10] + 20];
    f_[21] = f[off[15] + 21];
    f_[22] = f[off[17] + 22];
    f_[23] = f[off[16] + 23];
    f_[24] = f[off[12] + 24];
    f_[25] = f[off[14] + 25];
    f_[26] = f[off[13] + 26];

  } else {
    f_[0] = f[off[0] + 0];
    f_[1] = f[off[0] + 2];
    f_[2] = f[off[0] + 1];
    f_[3] = f[off[0] + 6];
    f_[4] = f[off[0] + 8];
    f_[5] = f[off[0] + 7];
    f_[6] = f[off[0] + 3];
    f_[7] = f[off[0] + 5];
    f_[8] = f[off[0] + 4];
    f_[9] = f[off[0] + 18];
    f_[10] = f[off[0] + 20];
    f_[11] = f[off[0] + 19];
    f_[12] = f[off[0] + 24];
    f_[13] = f[off[0] + 26];
    f_[14] = f[off[0] + 25];
    f_[15] = f[off[0] + 21];
    f_[16] = f[off[0] + 23];
    f_[17] = f[off[0] + 22];
    f_[18] = f[off[0] + 9];
    f_[19] = f[off[0] + 11];
    f_[20] = f[off[0] + 10];
    f_[21] = f[off[0] + 15];
    f_[22] = f[off[0] + 17];
    f_[23] = f[off[0] + 16];
    f_[24] = f[off[0] + 12];
    f_[25] = f[off[0] + 14];
    f_[26] = f[off[0] + 13];
  }

  // clang-format off
    float r = f_[0] + f_[1] + f_[2] + f_[3] + f_[4] + f_[5] + f_[6] + f_[7] + f_[8] + f_[9] + f_[10] + f_[11] + f_[12] + f_[13] + f_[14] + f_[15] + f_[16] + f_[17] + f_[18] + f_[19] + f_[20] + f_[21] + f_[22] + f_[23] + f_[24] + f_[25] + f_[26];
    float vx = ((f_[0] * 0)+(f_[1] * 0)+(f_[2] * 0)+(f_[3] * 0)+(f_[4] * 0)+(f_[5] * 0)+(f_[6] * 0)+(f_[7] * 0)+(f_[8] * 0)+(f_[9] * 1)+(f_[10] * 1)+(f_[11] * 1)+(f_[12] * 1)+(f_[13] * 1)+(f_[14] * 1)+(f_[15] * 1)+(f_[16] * 1)+(f_[17] * 1)+(f_[18] * -1)+(f_[19] * -1)+(f_[20] * -1)+(f_[21] * -1)+(f_[22] * -1)+(f_[23] * -1)+(f_[24] * -1)+(f_[25] * -1)+(f_[26] * -1)) / r;
    float vy = ((f_[0] * 0)+(f_[1] * 0)+(f_[2] * 0)+(f_[3] * 1)+(f_[4] * 1)+(f_[5] * 1)+(f_[6] * -1)+(f_[7] * -1)+(f_[8] * -1)+(f_[9] * 0)+(f_[10] * 0)+(f_[11] * 0)+(f_[12] * 1)+(f_[13] * 1)+(f_[14] * 1)+(f_[15] * -1)+(f_[16] * -1)+(f_[17] * -1)+(f_[18] * 0)+(f_[19] * 0)+(f_[20] * 0)+(f_[21] * 1)+(f_[22] * 1)+(f_[23] * 1)+(f_[24] * -1)+(f_[25] * -1)+(f_[26] * -1)) / r;
    float vz = ((f_[0] * 0)+(f_[1] * 1)+(f_[2] * -1)+(f_[3] * 0)+(f_[4] * 1)+(f_[5] * -1)+(f_[6] * 0)+(f_[7] * 1)+(f_[8] * -1)+(f_[9] * 0)+(f_[10] * 1)+(f_[11] * -1)+(f_[12] * 0)+(f_[13] * 1)+(f_[14] * -1)+(f_[15] * 0)+(f_[16] * 1)+(f_[17] * -1)+(f_[18] * 0)+(f_[19] * 1)+(f_[20] * -1)+(f_[21] * 0)+(f_[22] * 1)+(f_[23] * -1)+(f_[24] * 0)+(f_[25] * 1)+(f_[26] * -1)) / r;
  // clang-format on

  const float vv = vx * vx + vy * vy + vz * vz;
  vx += (gx / omega);
  vy += (gy / omega);
  vz += (gz / omega);

  if (fixed) {
    omega = 1.0;
    r = rho[ii];
    vx = vel[ii * 3 + 0];
    vy = vel[ii * 3 + 1];
    vz = vel[ii * 3 + 2];
  }
  // clang-format off
   f_[0] += omega * (r * (8/27) * (1 + 0 * vx + 0 * vy + 0 * vz + 0.5 * ((0 * vx + 0 * vy + 0 * vz) * (0 * vx + 0 * vy + 0 * vz) - vv)) - f_[0]);
   f_[1] += omega * (r * (2/27) * (1 + 0 * vx + 0 * vy + 1 * vz + 0.5 * ((0 * vx + 0 * vy + 1 * vz) * (0 * vx + 0 * vy + 1 * vz) - vv)) - f_[1]);
   f_[2] += omega * (r * (2/27) * (1 + 0 * vx + 0 * vy + -1 * vz + 0.5 * ((0 * vx + 0 * vy + -1 * vz) * (0 * vx + 0 * vy + -1 * vz) - vv)) - f_[2]);
   f_[3] += omega * (r * (2/27) * (1 + 0 * vx + 1 * vy + 0 * vz + 0.5 * ((0 * vx + 1 * vy + 0 * vz) * (0 * vx + 1 * vy + 0 * vz) - vv)) - f_[3]);
   f_[4] += omega * (r * (1/54) * (1 + 0 * vx + 1 * vy + 1 * vz + 0.5 * ((0 * vx + 1 * vy + 1 * vz) * (0 * vx + 1 * vy + 1 * vz) - vv)) - f_[4]);
   f_[5] += omega * (r * (1/54) * (1 + 0 * vx + 1 * vy + -1 * vz + 0.5 * ((0 * vx + 1 * vy + -1 * vz) * (0 * vx + 1 * vy + -1 * vz) - vv)) - f_[5]);
   f_[6] += omega * (r * (2/27) * (1 + 0 * vx + -1 * vy + 0 * vz + 0.5 * ((0 * vx + -1 * vy + 0 * vz) * (0 * vx + -1 * vy + 0 * vz) - vv)) - f_[6]);
   f_[7] += omega * (r * (1/54) * (1 + 0 * vx + -1 * vy + 1 * vz + 0.5 * ((0 * vx + -1 * vy + 1 * vz) * (0 * vx + -1 * vy + 1 * vz) - vv)) - f_[7]);
   f_[8] += omega * (r * (1/54) * (1 + 0 * vx + -1 * vy + -1 * vz + 0.5 * ((0 * vx + -1 * vy + -1 * vz) * (0 * vx + -1 * vy + -1 * vz) - vv)) - f_[8]);
   f_[9] += omega * (r * (2/27) * (1 + 1 * vx + 0 * vy + 0 * vz + 0.5 * ((1 * vx + 0 * vy + 0 * vz) * (1 * vx + 0 * vy + 0 * vz) - vv)) - f_[9]);
   f_[10] += omega * (r * (1/54) * (1 + 1 * vx + 0 * vy + 1 * vz + 0.5 * ((1 * vx + 0 * vy + 1 * vz) * (1 * vx + 0 * vy + 1 * vz) - vv)) - f_[10]);
   f_[11] += omega * (r * (1/54) * (1 + 1 * vx + 0 * vy + -1 * vz + 0.5 * ((1 * vx + 0 * vy + -1 * vz) * (1 * vx + 0 * vy + -1 * vz) - vv)) - f_[11]);
   f_[12] += omega * (r * (1/54) * (1 + 1 * vx + 1 * vy + 0 * vz + 0.5 * ((1 * vx + 1 * vy + 0 * vz) * (1 * vx + 1 * vy + 0 * vz) - vv)) - f_[12]);
   f_[13] += omega * (r * (1/216) * (1 + 1 * vx + 1 * vy + 1 * vz + 0.5 * ((1 * vx + 1 * vy + 1 * vz) * (1 * vx + 1 * vy + 1 * vz) - vv)) - f_[13]);
   f_[14] += omega * (r * (1/216) * (1 + 1 * vx + 1 * vy + -1 * vz + 0.5 * ((1 * vx + 1 * vy + -1 * vz) * (1 * vx + 1 * vy + -1 * vz) - vv)) - f_[14]);
   f_[15] += omega * (r * (1/54) * (1 + 1 * vx + -1 * vy + 0 * vz + 0.5 * ((1 * vx + -1 * vy + 0 * vz) * (1 * vx + -1 * vy + 0 * vz) - vv)) - f_[15]);
   f_[16] += omega * (r * (1/216) * (1 + 1 * vx + -1 * vy + 1 * vz + 0.5 * ((1 * vx + -1 * vy + 1 * vz) * (1 * vx + -1 * vy + 1 * vz) - vv)) - f_[16]);
   f_[17] += omega * (r * (1/216) * (1 + 1 * vx + -1 * vy + -1 * vz + 0.5 * ((1 * vx + -1 * vy + -1 * vz) * (1 * vx + -1 * vy + -1 * vz) - vv)) - f_[17]);
   f_[18] += omega * (r * (2/27) * (1 + -1 * vx + 0 * vy + 0 * vz + 0.5 * ((-1 * vx + 0 * vy + 0 * vz) * (-1 * vx + 0 * vy + 0 * vz) - vv)) - f_[18]);
   f_[19] += omega * (r * (1/54) * (1 + -1 * vx + 0 * vy + 1 * vz + 0.5 * ((-1 * vx + 0 * vy + 1 * vz) * (-1 * vx + 0 * vy + 1 * vz) - vv)) - f_[19]);
   f_[20] += omega * (r * (1/54) * (1 + -1 * vx + 0 * vy + -1 * vz + 0.5 * ((-1 * vx + 0 * vy + -1 * vz) * (-1 * vx + 0 * vy + -1 * vz) - vv)) - f_[20]);
   f_[21] += omega * (r * (1/54) * (1 + -1 * vx + 1 * vy + 0 * vz + 0.5 * ((-1 * vx + 1 * vy + 0 * vz) * (-1 * vx + 1 * vy + 0 * vz) - vv)) - f_[21]);
   f_[22] += omega * (r * (1/216) * (1 + -1 * vx + 1 * vy + 1 * vz + 0.5 * ((-1 * vx + 1 * vy + 1 * vz) * (-1 * vx + 1 * vy + 1 * vz) - vv)) - f_[22]);
   f_[23] += omega * (r * (1/216) * (1 + -1 * vx + 1 * vy + -1 * vz + 0.5 * ((-1 * vx + 1 * vy + -1 * vz) * (-1 * vx + 1 * vy + -1 * vz) - vv)) - f_[23]);
   f_[24] += omega * (r * (1/54) * (1 + -1 * vx + -1 * vy + 0 * vz + 0.5 * ((-1 * vx + -1 * vy + 0 * vz) * (-1 * vx + -1 * vy + 0 * vz) - vv)) - f_[24]);
   f_[25] += omega * (r * (1/216) * (1 + -1 * vx + -1 * vy + 1 * vz + 0.5 * ((-1 * vx + -1 * vy + 1 * vz) * (-1 * vx + -1 * vy + 1 * vz) - vv)) - f_[25]);
   f_[26] += omega * (r * (1/216) * (1 + -1 * vx + -1 * vy + -1 * vz + 0.5 * ((-1 * vx + -1 * vy + -1 * vz) * (-1 * vx + -1 * vy + -1 * vz) - vv)) - f_[26]);
  // clang-format on

  if (even) {
    f[0 + 0] = f_[0];
    f[2 + 1] = f_[1];
    f[1 + 2] = f_[2];
    f[6 + 3] = f_[3];
    f[8 + 4] = f_[4];
    f[7 + 5] = f_[5];
    f[3 + 6] = f_[6];
    f[5 + 7] = f_[7];
    f[4 + 8] = f_[8];
    f[18 + 9] = f_[9];
    f[20 + 10] = f_[10];
    f[19 + 11] = f_[11];
    f[24 + 12] = f_[12];
    f[26 + 13] = f_[13];
    f[25 + 14] = f_[14];
    f[21 + 15] = f_[15];
    f[23 + 16] = f_[16];
    f[22 + 17] = f_[17];
    f[9 + 18] = f_[18];
    f[11 + 19] = f_[19];
    f[10 + 20] = f_[20];
    f[15 + 21] = f_[21];
    f[17 + 22] = f_[22];
    f[16 + 23] = f_[23];
    f[12 + 24] = f_[24];
    f[14 + 25] = f_[25];
    f[13 + 26] = f_[26];

  } else {
    f[off[0] + 0] = f_[0];
    f[off[0] + 2] = f_[1];
    f[off[0] + 1] = f_[2];
    f[off[0] + 6] = f_[3];
    f[off[0] + 8] = f_[4];
    f[off[0] + 7] = f_[5];
    f[off[0] + 3] = f_[6];
    f[off[0] + 5] = f_[7];
    f[off[0] + 4] = f_[8];
    f[off[0] + 18] = f_[9];
    f[off[0] + 20] = f_[10];
    f[off[0] + 19] = f_[11];
    f[off[0] + 24] = f_[12];
    f[off[0] + 26] = f_[13];
    f[off[0] + 25] = f_[14];
    f[off[0] + 21] = f_[15];
    f[off[0] + 23] = f_[16];
    f[off[0] + 22] = f_[17];
    f[off[0] + 9] = f_[18];
    f[off[0] + 11] = f_[19];
    f[off[0] + 10] = f_[20];
    f[off[0] + 15] = f_[21];
    f[off[0] + 17] = f_[22];
    f[off[0] + 16] = f_[23];
    f[off[0] + 12] = f_[24];
    f[off[0] + 14] = f_[25];
    f[off[0] + 13] = f_[26];
  }

  rho[ii] = r;
  vel[ii * 3 + 0] = vx;
  vel[ii * 3 + 1] = vy;
  vel[ii * 3 + 2] = vz;
}