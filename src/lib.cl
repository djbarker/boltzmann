kernel void update_d2q9_bgk(int even, float omega, float gx, float gy, global float *f,
                            global float *rho, global float *vel,
                            global int *cell, global int *qs) {

  const size_t iy = get_global_id(0);
  const size_t ix = get_global_id(1);
  const size_t sy = get_global_size(0);
  const size_t sx = get_global_size(1);

  const size_t ii = ix * sy + iy;     // 1d idx for arrays
  // const size_t ii = iy * sx + ix;     // 1d idx for arrays

  const size_t iv = ii * 2;           // offset for vel
  const size_t if_ = ii * 9;          // offset for f

  const int c = cell[ii];
  const bool wall = (c & 1);
  const bool fixed = (c & 6); 

  if (wall) {
    // wall => do nothing
    return; 
  }

  // Array access conversion:
  //
  // rust:     idx[n]      -->  opencl:     idx[ii + n]
  // rust:   f[idx[n]][m]  -->  opencl:   f[idx[ii + n] * 9 + m]
  // rust: vel[idx[n]][m]  -->  opencl: vel[idx[ii + n] * 2 + m]
  // rust: rho[idx[n]]     -->  opencl: rho[idx[ii + n]]

  int off[8]; // do not bother with zero
  
  # pragma unroll
  for (int i=0; i<8; i++) {
    int ix_ = (ix + qs[2*(i+1) + 0] + sx) % sx;
    int iy_ = (iy + qs[2*(i+1) + 1] + sy) % sy;
    off[i] = ix_ * sy + iy_;
    // off[i] = iy_ * sx + ix_;
  }

  float f_[9];
  if (even) {
    f_[0] = f[if_ + 0];
    f_[1] = f[9 * off[1-1] + 1];
    f_[2] = f[9 * off[2-1] + 2];
    f_[3] = f[9 * off[3-1] + 3];
    f_[4] = f[9 * off[4-1] + 4];
    f_[5] = f[9 * off[5-1] + 5];
    f_[6] = f[9 * off[6-1] + 6];
    f_[7] = f[9 * off[7-1] + 7];
    f_[8] = f[9 * off[8-1] + 8];
  } else {
    f_[0] = f[if_ + 0];
    f_[1] = f[if_ + 2];
    f_[2] = f[if_ + 1];
    f_[3] = f[if_ + 4];
    f_[4] = f[if_ + 3];
    f_[5] = f[if_ + 6];
    f_[6] = f[if_ + 5];
    f_[7] = f[if_ + 8];
    f_[8] = f[if_ + 7];
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
    vx = vel[iv + 0];
    vy = vel[iv + 1];
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
    f[if_ + 0] = f_[0];
    f[9 * off[1 - 1] + 1] = f_[2];
    f[9 * off[2 - 1] + 2] = f_[1];
    f[9 * off[3 - 1] + 3] = f_[4];
    f[9 * off[4 - 1] + 4] = f_[3];
    f[9 * off[5 - 1] + 5] = f_[6];
    f[9 * off[6 - 1] + 6] = f_[5];
    f[9 * off[7 - 1] + 7] = f_[8];
    f[9 * off[8 - 1] + 8] = f_[7];
  } else {
    # pragma unroll
    for (int i = 0; i < 9; i++) {
      f[if_ + i] = f_[i];
    }
  }

  rho[ii] = r;
  vel[iv + 0] = vx;
  vel[iv + 1] = vy;
}

kernel void update_d2q5_bgk(int even, float omega, global float *f,
                            global float *val, global float *vel,
                            global int *cell, global int *qs) {

  const size_t iy = get_global_id(0);
  const size_t ix = get_global_id(1);
  const size_t sy = get_global_size(0);
  const size_t sx = get_global_size(1);

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

  int off[4]; // do not bother with zero
  
  # pragma unroll
  for (int i=0; i<4; i++) {
    int ix_ = (ix + qs[2*(i+1) + 0] + sx) % sx;
    int iy_ = (iy + qs[2*(i+1) + 1] + sy) % sy;
    off[i] = ix_ * sy + iy_;
  }

  float f_[5];
  if (even) {
    f_[0] = f[if_ + 0];
    f_[1] = f[5 * off[1 - 1] + 1];
    f_[2] = f[5 * off[2 - 1] + 2];
    f_[3] = f[5 * off[3 - 1] + 3];
    f_[4] = f[5 * off[4 - 1] + 4];
  } else {
    f_[0] = f[if_ + 0];
    f_[1] = f[if_ + 2];
    f_[2] = f[if_ + 1];
    f_[3] = f[if_ + 4];
    f_[4] = f[if_ + 3];
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
    f[if_ + 0] = f_[0];
    f[5 * off[1 - 1] + 1] = f_[2];
    f[5 * off[2 - 1] + 2] = f_[1];
    f[5 * off[3 - 1] + 3] = f_[4];
    f[5 * off[4 - 1] + 4] = f_[3];
  } else {
    # pragma unroll
    for (int i = 0; i < 5; i++) {
      f[if_ + i] = f_[i];
    }
  }

  val[ii] = C;
}
