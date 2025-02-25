kernel void update_d2q9_bgk(int even, float omega, float gx, float gy, global float *f,
                            global float *rho, global float *vel,
                            global int *cell, global int *qs, int sx, int sy) {

  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  // const size_t sx = get_global_size(0);
  // const size_t sy = get_global_size(1);

  if (ix >= sx || iy >= sy) return;

  const size_t ii = ix * sy + iy;     // 1d idx for arrays

  const int c = cell[ii];
  const bool wall = (c & 1);
  const bool fixed = (c & 6); 

  if (wall) {
    // wall => do nothing
    return; 
  }

  int off[9]; 
  
  # pragma unroll
  for (int i=0; i<9; i++) {
    int ix_ = (ix + qs[2*i + 0] + sx) % sx;
    int iy_ = (iy + qs[2*i + 1] + sy) % sy;
    off[i] = (ix_ * sy + iy_) * 9;
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

  vx += (gx / omega);
  vy += (gy / omega);

  if (fixed) {
    omega = 1.0;
    r = rho[ii];
    vx = vel[ii*2 + 0];
    vy = vel[ii*2 + 1];
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
    # pragma unroll
    for (int i = 0; i < 9; i++) {
      f[off[0] + i] = f_[i];
    }
  }

  rho[ii] = r;
  vel[ii*2 + 0] = vx;
  vel[ii*2 + 1] = vy;
}

kernel void update_d2q5_bgk(int even, float omega, global float *f,
                            global float *val, global float *vel,
                            global int *cell, global int *qs, int sx, int sy) {

  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  // const size_t sx = get_global_size(0);
  // const size_t sy = get_global_size(1);

  if (ix >= sx || iy >= sy) return;


  const size_t ii = ix * sy + iy;     // 1d idx for arrays

  const size_t iv = ii * 2;           // offset for vel
  const size_t if_ = ii * 5;          // offset for f

  const int c = cell[ii];
  const bool wall = (c & 1); 
  const bool fixed = (c & 8);

  if (wall) {
    // wall => do nothing
    return;
  }

  int off[5];
  
  # pragma unroll
  for (int i=0; i<5; i++) {
    int ix_ = (ix + qs[2*i + 0] + sx) % sx;
    int iy_ = (iy + qs[2*i + 1] + sy) % sy;
    off[i] = (ix_ * sy + iy_) * 5;
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
    # pragma unroll
    for (int i = 0; i < 5; i++) {
      f[off[0] + i] = f_[i];
    }
  }

  val[ii] = C;
}
