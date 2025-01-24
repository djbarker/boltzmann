kernel void update_d2q9_bgk(int even, float omega, global float *f,
                            global float *rho, global float *vel,
                            global int *cell, global int *idx) {

  const size_t ir = get_global_id(0); // offset for rho
  const size_t ii = ir * 9;           // offset for idx
  const size_t iv = ir * 2;           // offset for vel

  const int c = cell[ir];
  const bool wall = c == 1;
  const bool fixed = c == 2;

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

  float f_[9];
  if (even) {
    f_[0] = f[9 * idx[ii + 0] + 0];
    f_[1] = f[9 * idx[ii + 1] + 1];
    f_[2] = f[9 * idx[ii + 2] + 2];
    f_[3] = f[9 * idx[ii + 3] + 3];
    f_[4] = f[9 * idx[ii + 4] + 4];
    f_[5] = f[9 * idx[ii + 5] + 5];
    f_[6] = f[9 * idx[ii + 6] + 6];
    f_[7] = f[9 * idx[ii + 7] + 7];
    f_[8] = f[9 * idx[ii + 8] + 8];
  } else {
    f_[0] = f[9 * idx[ii + 0] + 0];
    f_[1] = f[9 * idx[ii + 0] + 2];
    f_[2] = f[9 * idx[ii + 0] + 1];
    f_[3] = f[9 * idx[ii + 0] + 4];
    f_[4] = f[9 * idx[ii + 0] + 3];
    f_[5] = f[9 * idx[ii + 0] + 6];
    f_[6] = f[9 * idx[ii + 0] + 5];
    f_[7] = f[9 * idx[ii + 0] + 8];
    f_[8] = f[9 * idx[ii + 0] + 7];
  }

  // calc moments
  float r =
      f_[0] + f_[1] + f_[2] + f_[3] + f_[4] + f_[5] + f_[6] + f_[7] + f_[8];
  float vx = (f_[1] - f_[2] + f_[5] - f_[6] - f_[7] + f_[8]) / r;
  float vy = (f_[3] - f_[4] + f_[5] - f_[6] + f_[7] - f_[8]) / r;

  if (fixed) {
    omega = 1.0;
    r = rho[ir];
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
    f[9 * idx[ii + 0] + 0] = f_[0];
    f[9 * idx[ii + 1] + 1] = f_[2];
    f[9 * idx[ii + 2] + 2] = f_[1];
    f[9 * idx[ii + 3] + 3] = f_[4];
    f[9 * idx[ii + 4] + 4] = f_[3];
    f[9 * idx[ii + 5] + 5] = f_[6];
    f[9 * idx[ii + 6] + 6] = f_[5];
    f[9 * idx[ii + 7] + 7] = f_[8];
    f[9 * idx[ii + 8] + 8] = f_[7];
  } else {
    for (int i = 0; i < 9; i++) {
      f[9 * idx[ii + 0] + i] = f_[i];
    }
  }

  // TODO: we don't actually need to update this until the end of loop_for
  if (!fixed) {
    rho[ir] = r;
    vel[2 * idx[ii + 0] + 0] = vx;
    vel[2 * idx[ii + 0] + 1] = vy;
  }
}
