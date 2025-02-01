use std::ops::{Add, Rem};

use ndarray::{Array, DimAdd, Dimension};

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
pub fn vmod<const D: usize, T>(x: &VectS<T, D>, m: &VectS<T, D>) -> VectS<T, D>
where
    T: Rem<Output = T> + Add<Output = T> + Copy + Default + 'static,
{
    x.map_with_idx(|j, x| fmod(x, m[j]))
}

/// Elementwise modulo the components of `x` with those of `m` using [`fmod`].  
pub fn vmod_nd<T, D: Dimension>(x: Array<T, D>, m: &Array<T, D>) -> Array<T, D>
where
    T: Add<T, Output = T> + Rem<Output = T> + Copy,
{
    let mut out = x.clone();
    out.zip_mut_with(m, |xx, mm| *xx = fmod(*xx, *mm));
    out
}
