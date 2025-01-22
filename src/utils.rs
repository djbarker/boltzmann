use std::ops::{Add, Rem};

use crate::vect_s::VectS;

// Split a (sufficiently small & positive) float into its integer and fractional parts.
pub fn split_int_frac(x: f32) -> (i32, f32) {
    let i = x.floor();
    let f = x - i;
    (i as i32, f)
}

/// Modulo operation which handles -ve `x`.
pub fn fmod<T>(x: T, m: T) -> T
where
    T: Rem<Output = T> + Add<Output = T> + Copy,
{
    (x + m) % m
}

/// Elementwise modulo the components of `x` with those of `m` using [`fmod`].  
pub fn vmod<const D: usize, T>(x: VectS<T, D>, m: VectS<T, D>) -> VectS<T, D>
where
    T: Rem<Output = T> + Add<Output = T> + Copy + Default + 'static,
{
    x.map_with_idx(|j, x| fmod(x, m[j]))
}

/// Calculate the domain decomposition counts in each dimension.
///
/// The decomposition is always such that subdomain cuboids have their verticies touching,
/// edges touching, etc.
/// It tries to keep the subdomains as close to cubic as possible given this constraint.
pub fn decomp_2d(ncpus: usize) -> [usize; 2] {
    let mut root = 1;
    for i in 2..=ncpus.sqrt() {
        if ncpus % i == 0 {
            root = i;
        }
    }

    [root, ncpus / root]
}


/// Calculate the domain decomposition counts in each dimension.
///
/// The decomposition is always such that subdomain cuboids have their verticies touching,
/// edges touching, etc.
/// It tries to keep the subdomains as close to cubic as possible given this constraint.
pub fn decomp_nd<const N: usize>(ncpus: usize) -> [usize; N] {
    if N == 1 {
        return [ncpus];
    }

    let mut root = 1;
    let imax = (ncpus as f32).pow(1.0 / (N as f32)) as usize;
    for i in 2..=imax {
        if ncpus % i == 0 {
            root = i;
        }
    }

    [[root], decomp_nd<N-1>(ncpus/root)].concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decomp_2d() {
        assert_eq!(decomp_2d(1), [1, 1]);
        assert_eq!(decomp_2d(2), [1, 2]);
        assert_eq!(decomp_2d(3), [1, 3]);
        assert_eq!(decomp_2d(4), [2, 2]);
        assert_eq!(decomp_2d(5), [1, 5]);
        assert_eq!(decomp_2d(6), [2, 3]);
        assert_eq!(decomp_2d(7), [1, 7]);
        assert_eq!(decomp_2d(8), [2, 4]);
        assert_eq!(decomp_2d(9), [3, 3]);
        assert_eq!(decomp_2d(10), [2, 5]);
    }
    
    #[test]
    fn test_decomp_3d() {
        assert_eq!(decomp_nd(1), [1, 1, 1]);
        assert_eq!(decomp_nd(2), [1, 1, 2]);
        assert_eq!(decomp_nd(3), [1, 1, 3]);
        assert_eq!(decomp_nd(4), [1, 2, 2]);
        assert_eq!(decomp_nd(5), [1, 1, 5]);
        assert_eq!(decomp_nd(6), [1, 2, 3]);
        assert_eq!(decomp_nd(7), [1, 1, 7]);
        assert_eq!(decomp_nd(8), [2, 2, 2]);
        assert_eq!(decomp_nd(9), [1, 3, 3]);
        assert_eq!(decomp_nd(10), [1, 2, 5]);
    }

    #[test]
    fn test_decomp_nd() {
        assert_eq!(decomp_2d(1), decomp_nd(1));
        assert_eq!(decomp_2d(2), decomp_nd(2));
        assert_eq!(decomp_2d(3), decomp_nd(3));
        assert_eq!(decomp_2d(4), decomp_nd(4));
        assert_eq!(decomp_2d(5), decomp_nd(5));
        assert_eq!(decomp_2d(6), decomp_nd(6));
        assert_eq!(decomp_2d(7), decomp_nd(7));
        assert_eq!(decomp_2d(8), decomp_nd(8));
        assert_eq!(decomp_2d(9), decomp_nd(9));
        assert_eq!(decomp_2d(10), decomp_nd(10));
    }
}
