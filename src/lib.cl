kernel void update_d2q9_bgk_handrolled(__constant int *s, int even, float omega,
                                       __constant float *g, global float *f,
                                       global float *rho, global float *vel,
                                       global int *cell) {

  const int qs[9][2] = {{0, 0}, {1, 0},   {-1, 0}, {0, 1}, {0, -1},
                        {1, 1}, {-1, -1}, {-1, 1}, {1, -1}};

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

kernel void update_d2q5_bgk_handrolled(__constant int *s, int even, float omega,
                                       global float *f, global float *val,
                                       global float *vel, global int *cell) {
  const int qs[5][2] = {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};

  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  // const size_t sx = get_global_size(0);
  // const size_t sy = get_global_size(1);

  if (ix >= s[0] || iy >= s[1])
    return;

  const size_t ii = ix * s[1] + iy; // 1d idx for arrays

  const size_t iv = ii * 2; // offset for vel

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

kernel void set_constant_acc_2d(__constant int *s, global float *acc,
                                global float *g) {

  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);

  if (ix >= s[0] || iy >= s[1])
    return;

  const size_t ii = ix * s[1] + iy;

  acc[ii * 2 + 0] = g[0];
  acc[ii * 2 + 1] = g[1];
}

kernel void set_boussinesq_acc_2d(__constant int *s, global float *acc,
                                  global float *conc, float alpha, float c0,
                                  global float *g) {
  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);

  if (ix >= s[0] || iy >= s[1])
    return;

  const size_t ii = ix * s[1] + iy;

  const float buoyancy = alpha * (conc[ii] - c0);
  acc[ii * 2 + 0] = g[0] * buoyancy;
  acc[ii * 2 + 1] = g[1] * buoyancy;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
//                             AUTO GENERATED KERNELS FOLLOW
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

kernel void update_d2q5_bgk(__constant int *s, int even, float omega,
                            global float *f, global float *val,
                            global float *vel, global int *cell) {
  const int qs[5][2] = {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};

  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);

  if (ix >= s[0] || iy >= s[1])
    return;

  const size_t ii = ix * s[1] + iy;

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
    const int ix_ = (ix + qs[i][0] + s[0]) % s[0];
    const int iy_ = (iy + qs[i][1] + s[1]) % s[1];

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

  // clang-format off
   float r = f_[0] + f_[1] + f_[2] + f_[3] + f_[4];
   float vx = vel[2 * ii + 0];
   float vy = vel[2 * ii + 1];
  // clang-format on

  if (fixed) {
    omega = 1.0;
    r = val[ii];
    vx = vel[ii * 2 + 0];
    vy = vel[ii * 2 + 1];
  }

  const float vv = vx * vx + vy * vy;

  // clang-format off
   f_[0] += omega * (r * (1.0/3.0 - 1.0/2.0*vv) - f_[0]);
   f_[1] += omega * (r * (-1.0/4.0*vv + (3.0/4.0)*pow(vx, 2) + (1.0/2.0)*vx + 1.0/6.0) - f_[1]);
   f_[2] += omega * (r * (-1.0/4.0*vv + (3.0/4.0)*pow(vx, 2) - 1.0/2.0*vx + 1.0/6.0) - f_[2]);
   f_[3] += omega * (r * (-1.0/4.0*vv + (3.0/4.0)*pow(vy, 2) + (1.0/2.0)*vy + 1.0/6.0) - f_[3]);
   f_[4] += omega * (r * (-1.0/4.0*vv + (3.0/4.0)*pow(vy, 2) - 1.0/2.0*vy + 1.0/6.0) - f_[4]);
  // clang-format on

  if (even) {
    f[off[0] + 0] = f_[0];
    f[off[2] + 1] = f_[2];
    f[off[1] + 2] = f_[1];
    f[off[4] + 3] = f_[4];
    f[off[3] + 4] = f_[3];

  } else {
    f[off[0] + 0] = f_[0];
    f[off[0] + 1] = f_[1];
    f[off[0] + 2] = f_[2];
    f[off[0] + 3] = f_[3];
    f[off[0] + 4] = f_[4];
  }

  val[ii] = r;
}

kernel void update_d2q9_bgk(__constant int *s, int even, float omega,
                            global float *f, global float *rho,
                            global float *vel, global float *acc, int use_acc,
                            global int *cell) {
  const int qs[9][2] = {{0, 0}, {1, 0},   {-1, 0}, {0, 1}, {0, -1},
                        {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};

  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);

  if (ix >= s[0] || iy >= s[1])
    return;

  const size_t ii = ix * s[1] + iy;

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
    const int ix_ = (ix + qs[i][0] + s[0]) % s[0];
    const int iy_ = (iy + qs[i][1] + s[1]) % s[1];

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

  // clang-format off
   float r = f_[0] + f_[1] + f_[2] + f_[3] + f_[4] + f_[5] + f_[6] + f_[7] + f_[8];
   float vx = (f_[1] - f_[2] + f_[5] - f_[6] + f_[7] - f_[8]) / r;
   float vy = (f_[3] - f_[4] + f_[5] - f_[6] - f_[7] + f_[8]) / r;
  // clang-format on

  if (use_acc) {
    vx += acc[ii * 2 + 0] / omega;
    vy += acc[ii * 2 + 1] / omega;
  }

  if (fixed) {
    omega = 1.0;
    r = rho[ii];
    vx = vel[ii * 2 + 0];
    vy = vel[ii * 2 + 1];
  }

  const float vv = vx * vx + vy * vy;

  // clang-format off
   f_[0] += omega * (r * (4.0/9.0 - 2.0/3.0*vv) - f_[0]);
   f_[1] += omega * (r * (-1.0/6.0*vv + (1.0/2.0)*pow(vx, 2) + (1.0/3.0)*vx + 1.0/9.0) - f_[1]);
   f_[2] += omega * (r * (-1.0/6.0*vv + (1.0/2.0)*pow(vx, 2) - 1.0/3.0*vx + 1.0/9.0) - f_[2]);
   f_[3] += omega * (r * (-1.0/6.0*vv + (1.0/2.0)*pow(vy, 2) + (1.0/3.0)*vy + 1.0/9.0) - f_[3]);
   f_[4] += omega * (r * (-1.0/6.0*vv + (1.0/2.0)*pow(vy, 2) - 1.0/3.0*vy + 1.0/9.0) - f_[4]);
   f_[5] += omega * (r * (-1.0/24.0*vv + (1.0/12.0)*vx + (1.0/12.0)*vy + (1.0/8.0)*pow(vx + vy, 2) + 1.0/36.0) - f_[5]);
   f_[6] += omega * (r * (-1.0/24.0*vv - 1.0/12.0*vx - 1.0/12.0*vy + (1.0/8.0)*pow(vx + vy, 2) + 1.0/36.0) - f_[6]);
   f_[7] += omega * (r * (-1.0/24.0*vv + (1.0/12.0)*vx - 1.0/12.0*vy + (1.0/8.0)*pow(vx - vy, 2) + 1.0/36.0) - f_[7]);
   f_[8] += omega * (r * (-1.0/24.0*vv - 1.0/12.0*vx + (1.0/12.0)*vy + (1.0/8.0)*pow(vx - vy, 2) + 1.0/36.0) - f_[8]);
  // clang-format on

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
    f[off[0] + 0] = f_[0];
    f[off[0] + 1] = f_[1];
    f[off[0] + 2] = f_[2];
    f[off[0] + 3] = f_[3];
    f[off[0] + 4] = f_[4];
    f[off[0] + 5] = f_[5];
    f[off[0] + 6] = f_[6];
    f[off[0] + 7] = f_[7];
    f[off[0] + 8] = f_[8];
  }

  rho[ii] = r;
  vel[ii * 2 + 0] = vx;
  vel[ii * 2 + 1] = vy;
}

kernel void update_d3q7_bgk(__constant int *s, int even, float omega,
                            global float *f, global float *val,
                            global float *vel, global int *cell) {
  const int qs[7][3] = {{0, 0, 0},  {1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                        {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  const size_t iz = get_global_id(2);

  if (ix >= s[0] || iy >= s[1] || iz >= s[2])
    return;

  const size_t ii = ix * s[1] * s[2] + iy * s[2] + iz;

  const int c = cell[ii];
  const bool wall = (c & 1);
  const bool fixed = (c & 8);

  if (wall) {
    // wall => do nothing
    return;
  }

  int off[7];

#pragma unroll
  for (int i = 0; i < 7; i++) {
    const int ix_ = (ix + qs[i][0] + s[0]) % s[0];
    const int iy_ = (iy + qs[i][1] + s[1]) % s[1];
    const int iz_ = (iz + qs[i][2] + s[2]) % s[2];

    off[i] = (iz_ + s[2] * (ix_ * s[1] + iy_)) * 7;
  }

  float f_[7];
  if (even) {
    f_[0] = f[off[0] + 0];
    f_[1] = f[off[2] + 1];
    f_[2] = f[off[1] + 2];
    f_[3] = f[off[4] + 3];
    f_[4] = f[off[3] + 4];
    f_[5] = f[off[6] + 5];
    f_[6] = f[off[5] + 6];

  } else {
    f_[0] = f[off[0] + 0];
    f_[1] = f[off[0] + 2];
    f_[2] = f[off[0] + 1];
    f_[3] = f[off[0] + 4];
    f_[4] = f[off[0] + 3];
    f_[5] = f[off[0] + 6];
    f_[6] = f[off[0] + 5];
  }

  // clang-format off
   float r = f_[0] + f_[1] + f_[2] + f_[3] + f_[4] + f_[5] + f_[6];
   float vx = vel[3 * ii + 0];
   float vy = vel[3 * ii + 1];
   float vz = vel[3 * ii + 2];
  // clang-format on

  if (fixed) {
    omega = 1.0;
    r = val[ii];
    vx = vel[ii * 3 + 0];
    vy = vel[ii * 3 + 1];
    vz = vel[ii * 3 + 2];
  }

  const float vv = vx * vx + vy * vy + vz * vz;

  // clang-format off
   f_[0] += omega * (r * (1.0/20.0 - 3.0/40.0*vv) - f_[0]);
   f_[1] += omega * (r * (-19.0/80.0*vv + (57.0/80.0)*pow(vx, 2) + (19.0/40.0)*vx + 19.0/120.0) - f_[1]);
   f_[2] += omega * (r * (-19.0/80.0*vv + (57.0/80.0)*pow(vx, 2) - 19.0/40.0*vx + 19.0/120.0) - f_[2]);
   f_[3] += omega * (r * (-19.0/80.0*vv + (57.0/80.0)*pow(vy, 2) + (19.0/40.0)*vy + 19.0/120.0) - f_[3]);
   f_[4] += omega * (r * (-19.0/80.0*vv + (57.0/80.0)*pow(vy, 2) - 19.0/40.0*vy + 19.0/120.0) - f_[4]);
   f_[5] += omega * (r * (-19.0/80.0*vv + (57.0/80.0)*pow(vz, 2) + (19.0/40.0)*vz + 19.0/120.0) - f_[5]);
   f_[6] += omega * (r * (-19.0/80.0*vv + (57.0/80.0)*pow(vz, 2) - 19.0/40.0*vz + 19.0/120.0) - f_[6]);
  // clang-format on

  if (even) {
    f[off[0] + 0] = f_[0];
    f[off[2] + 1] = f_[2];
    f[off[1] + 2] = f_[1];
    f[off[4] + 3] = f_[4];
    f[off[3] + 4] = f_[3];
    f[off[6] + 5] = f_[6];
    f[off[5] + 6] = f_[5];

  } else {
    f[off[0] + 0] = f_[0];
    f[off[0] + 1] = f_[1];
    f[off[0] + 2] = f_[2];
    f[off[0] + 3] = f_[3];
    f[off[0] + 4] = f_[4];
    f[off[0] + 5] = f_[5];
    f[off[0] + 6] = f_[6];
  }

  val[ii] = r;
}

kernel void update_d3q27_bgk(__constant int *s, int even, float omega,
                             __constant float *g, global float *f,
                             global float *rho, global float *vel,
                             global int *cell) {
  const int qs[27][3] = {{0, 0, 0},    {1, 0, 0},   {-1, 0, 0},  {0, 1, 0},
                         {0, -1, 0},   {0, 0, 1},   {0, 0, -1},  {1, 1, 0},
                         {-1, -1, 0},  {1, -1, 0},  {-1, 1, 0},  {1, 0, 1},
                         {-1, 0, -1},  {0, 1, 1},   {0, -1, -1}, {1, 0, -1},
                         {-1, 0, 1},   {0, 1, -1},  {0, -1, 1},  {1, 1, 1},
                         {-1, -1, -1}, {1, -1, 1},  {-1, 1, -1}, {1, 1, -1},
                         {-1, -1, 1},  {1, -1, -1}, {-1, 1, 1}};

  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  const size_t iz = get_global_id(2);

  if (ix >= s[0] || iy >= s[1] || iz >= s[2])
    return;

  const size_t ii = ix * s[1] * s[2] + iy * s[2] + iz;

  const int c = cell[ii];
  const bool wall = (c & 1);
  const bool fixed = (c & 6);

  if (wall) {
    // wall => do nothing
    return;
  }

  int off[27];

#pragma unroll
  for (int i = 0; i < 27; i++) {
    const int ix_ = (ix + qs[i][0] + s[0]) % s[0];
    const int iy_ = (iy + qs[i][1] + s[1]) % s[1];
    const int iz_ = (iz + qs[i][2] + s[2]) % s[2];

    off[i] = (iz_ + s[2] * (ix_ * s[1] + iy_)) * 27;
  }

  float f_[27];
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
    f_[9] = f[off[10] + 9];
    f_[10] = f[off[9] + 10];
    f_[11] = f[off[12] + 11];
    f_[12] = f[off[11] + 12];
    f_[13] = f[off[14] + 13];
    f_[14] = f[off[13] + 14];
    f_[15] = f[off[16] + 15];
    f_[16] = f[off[15] + 16];
    f_[17] = f[off[18] + 17];
    f_[18] = f[off[17] + 18];
    f_[19] = f[off[20] + 19];
    f_[20] = f[off[19] + 20];
    f_[21] = f[off[22] + 21];
    f_[22] = f[off[21] + 22];
    f_[23] = f[off[24] + 23];
    f_[24] = f[off[23] + 24];
    f_[25] = f[off[26] + 25];
    f_[26] = f[off[25] + 26];

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
    f_[9] = f[off[0] + 10];
    f_[10] = f[off[0] + 9];
    f_[11] = f[off[0] + 12];
    f_[12] = f[off[0] + 11];
    f_[13] = f[off[0] + 14];
    f_[14] = f[off[0] + 13];
    f_[15] = f[off[0] + 16];
    f_[16] = f[off[0] + 15];
    f_[17] = f[off[0] + 18];
    f_[18] = f[off[0] + 17];
    f_[19] = f[off[0] + 20];
    f_[20] = f[off[0] + 19];
    f_[21] = f[off[0] + 22];
    f_[22] = f[off[0] + 21];
    f_[23] = f[off[0] + 24];
    f_[24] = f[off[0] + 23];
    f_[25] = f[off[0] + 26];
    f_[26] = f[off[0] + 25];
  }

  // clang-format off
   float r = f_[0] + f_[1] + f_[2] + f_[3] + f_[4] + f_[5] + f_[6] + f_[7] + f_[8] + f_[9] + f_[10] + f_[11] + f_[12] + f_[13] + f_[14] + f_[15] + f_[16] + f_[17] + f_[18] + f_[19] + f_[20] + f_[21] + f_[22] + f_[23] + f_[24] + f_[25] + f_[26];
   float vx = (-f_[10] + f_[11] - f_[12] + f_[15] - f_[16] + f_[19] + f_[1] - f_[20] + f_[21] - f_[22] + f_[23] - f_[24] + f_[25] - f_[26] - f_[2] + f_[7] - f_[8] + f_[9]) / r;
   float vy = (f_[10] + f_[13] - f_[14] + f_[17] - f_[18] + f_[19] - f_[20] - f_[21] + f_[22] + f_[23] - f_[24] - f_[25] + f_[26] + f_[3] - f_[4] + f_[7] - f_[8] - f_[9]) / r;
   float vz = (f_[11] - f_[12] + f_[13] - f_[14] - f_[15] + f_[16] - f_[17] + f_[18] + f_[19] - f_[20] + f_[21] - f_[22] - f_[23] + f_[24] - f_[25] + f_[26] + f_[5] - f_[6]) / r;
  // clang-format on
  vx += (g[0] / omega);
  vy += (g[1] / omega);
  vz += (g[2] / omega);

  if (fixed) {
    omega = 1.0;
    r = rho[ii];
    vx = vel[ii * 3 + 0];
    vy = vel[ii * 3 + 1];
    vz = vel[ii * 3 + 2];
  }

  const float vv = vx * vx + vy * vy + vz * vz;

  // clang-format off
   f_[0] += omega * (r * (8.0/27.0 - 4.0/9.0*vv) - f_[0]);
   f_[1] += omega * (r * (-1.0/9.0*vv + (1.0/3.0)*pow(vx, 2) + (2.0/9.0)*vx + 2.0/27.0) - f_[1]);
   f_[2] += omega * (r * (-1.0/9.0*vv + (1.0/3.0)*pow(vx, 2) - 2.0/9.0*vx + 2.0/27.0) - f_[2]);
   f_[3] += omega * (r * (-1.0/9.0*vv + (1.0/3.0)*pow(vy, 2) + (2.0/9.0)*vy + 2.0/27.0) - f_[3]);
   f_[4] += omega * (r * (-1.0/9.0*vv + (1.0/3.0)*pow(vy, 2) - 2.0/9.0*vy + 2.0/27.0) - f_[4]);
   f_[5] += omega * (r * (-1.0/9.0*vv + (1.0/3.0)*pow(vz, 2) + (2.0/9.0)*vz + 2.0/27.0) - f_[5]);
   f_[6] += omega * (r * (-1.0/9.0*vv + (1.0/3.0)*pow(vz, 2) - 2.0/9.0*vz + 2.0/27.0) - f_[6]);
   f_[7] += omega * (r * (-1.0/36.0*vv + (1.0/18.0)*vx + (1.0/18.0)*vy + (1.0/12.0)*pow(vx + vy, 2) + 1.0/54.0) - f_[7]);
   f_[8] += omega * (r * (-1.0/36.0*vv - 1.0/18.0*vx - 1.0/18.0*vy + (1.0/12.0)*pow(vx + vy, 2) + 1.0/54.0) - f_[8]);
   f_[9] += omega * (r * (-1.0/36.0*vv + (1.0/18.0)*vx - 1.0/18.0*vy + (1.0/12.0)*pow(vx - vy, 2) + 1.0/54.0) - f_[9]);
   f_[10] += omega * (r * (-1.0/36.0*vv - 1.0/18.0*vx + (1.0/18.0)*vy + (1.0/12.0)*pow(vx - vy, 2) + 1.0/54.0) - f_[10]);
   f_[11] += omega * (r * (-1.0/36.0*vv + (1.0/18.0)*vx + (1.0/18.0)*vz + (1.0/12.0)*pow(vx + vz, 2) + 1.0/54.0) - f_[11]);
   f_[12] += omega * (r * (-1.0/36.0*vv - 1.0/18.0*vx - 1.0/18.0*vz + (1.0/12.0)*pow(vx + vz, 2) + 1.0/54.0) - f_[12]);
   f_[13] += omega * (r * (-1.0/36.0*vv + (1.0/18.0)*vy + (1.0/18.0)*vz + (1.0/12.0)*pow(vy + vz, 2) + 1.0/54.0) - f_[13]);
   f_[14] += omega * (r * (-1.0/36.0*vv - 1.0/18.0*vy - 1.0/18.0*vz + (1.0/12.0)*pow(vy + vz, 2) + 1.0/54.0) - f_[14]);
   f_[15] += omega * (r * (-1.0/36.0*vv + (1.0/18.0)*vx - 1.0/18.0*vz + (1.0/12.0)*pow(vx - vz, 2) + 1.0/54.0) - f_[15]);
   f_[16] += omega * (r * (-1.0/36.0*vv - 1.0/18.0*vx + (1.0/18.0)*vz + (1.0/12.0)*pow(vx - vz, 2) + 1.0/54.0) - f_[16]);
   f_[17] += omega * (r * (-1.0/36.0*vv + (1.0/18.0)*vy - 1.0/18.0*vz + (1.0/12.0)*pow(vy - vz, 2) + 1.0/54.0) - f_[17]);
   f_[18] += omega * (r * (-1.0/36.0*vv - 1.0/18.0*vy + (1.0/18.0)*vz + (1.0/12.0)*pow(vy - vz, 2) + 1.0/54.0) - f_[18]);
   f_[19] += omega * (r * (-1.0/144.0*vv + (1.0/72.0)*vx + (1.0/72.0)*vy + (1.0/72.0)*vz + (1.0/48.0)*pow(vx + vy + vz, 2) + 1.0/216.0) - f_[19]);
   f_[20] += omega * (r * (-1.0/144.0*vv - 1.0/72.0*vx - 1.0/72.0*vy - 1.0/72.0*vz + (1.0/48.0)*pow(vx + vy + vz, 2) + 1.0/216.0) - f_[20]);
   f_[21] += omega * (r * (-1.0/144.0*vv + (1.0/72.0)*vx - 1.0/72.0*vy + (1.0/72.0)*vz + (1.0/48.0)*pow(vx - vy + vz, 2) + 1.0/216.0) - f_[21]);
   f_[22] += omega * (r * (-1.0/144.0*vv - 1.0/72.0*vx + (1.0/72.0)*vy - 1.0/72.0*vz + (1.0/48.0)*pow(vx - vy + vz, 2) + 1.0/216.0) - f_[22]);
   f_[23] += omega * (r * (-1.0/144.0*vv + (1.0/72.0)*vx + (1.0/72.0)*vy - 1.0/72.0*vz + (1.0/48.0)*pow(vx + vy - vz, 2) + 1.0/216.0) - f_[23]);
   f_[24] += omega * (r * (-1.0/144.0*vv - 1.0/72.0*vx - 1.0/72.0*vy + (1.0/72.0)*vz + (1.0/48.0)*pow(vx + vy - vz, 2) + 1.0/216.0) - f_[24]);
   f_[25] += omega * (r * (-1.0/144.0*vv + (1.0/72.0)*vx - 1.0/72.0*vy - 1.0/72.0*vz + (1.0/48.0)*pow(-vx + vy + vz, 2) + 1.0/216.0) - f_[25]);
   f_[26] += omega * (r * (-1.0/144.0*vv - 1.0/72.0*vx + (1.0/72.0)*vy + (1.0/72.0)*vz + (1.0/48.0)*pow(-vx + vy + vz, 2) + 1.0/216.0) - f_[26]);
  // clang-format on

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
    f[off[10] + 9] = f_[10];
    f[off[9] + 10] = f_[9];
    f[off[12] + 11] = f_[12];
    f[off[11] + 12] = f_[11];
    f[off[14] + 13] = f_[14];
    f[off[13] + 14] = f_[13];
    f[off[16] + 15] = f_[16];
    f[off[15] + 16] = f_[15];
    f[off[18] + 17] = f_[18];
    f[off[17] + 18] = f_[17];
    f[off[20] + 19] = f_[20];
    f[off[19] + 20] = f_[19];
    f[off[22] + 21] = f_[22];
    f[off[21] + 22] = f_[21];
    f[off[24] + 23] = f_[24];
    f[off[23] + 24] = f_[23];
    f[off[26] + 25] = f_[26];
    f[off[25] + 26] = f_[25];

  } else {
    f[off[0] + 0] = f_[0];
    f[off[0] + 1] = f_[1];
    f[off[0] + 2] = f_[2];
    f[off[0] + 3] = f_[3];
    f[off[0] + 4] = f_[4];
    f[off[0] + 5] = f_[5];
    f[off[0] + 6] = f_[6];
    f[off[0] + 7] = f_[7];
    f[off[0] + 8] = f_[8];
    f[off[0] + 9] = f_[9];
    f[off[0] + 10] = f_[10];
    f[off[0] + 11] = f_[11];
    f[off[0] + 12] = f_[12];
    f[off[0] + 13] = f_[13];
    f[off[0] + 14] = f_[14];
    f[off[0] + 15] = f_[15];
    f[off[0] + 16] = f_[16];
    f[off[0] + 17] = f_[17];
    f[off[0] + 18] = f_[18];
    f[off[0] + 19] = f_[19];
    f[off[0] + 20] = f_[20];
    f[off[0] + 21] = f_[21];
    f[off[0] + 22] = f_[22];
    f[off[0] + 23] = f_[23];
    f[off[0] + 24] = f_[24];
    f[off[0] + 25] = f_[25];
    f[off[0] + 26] = f_[26];
  }

  rho[ii] = r;
  vel[ii * 3 + 0] = vx;
  vel[ii * 3 + 1] = vy;
  vel[ii * 3 + 2] = vz;
}
