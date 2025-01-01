use std::{
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    ptr::slice_from_raw_parts,
};

use num_traits::{AsPrimitive, One, Zero};
use numpy::{
    npyffi::NPY_TYPES, PyArrayDescr, PyArrayMethods, PyReadwriteArray1, PyUntypedArrayMethods,
};

/// A statically sized vector (i.e. size known at compile time)
/// This is a thin wrapper around an array which provides some convenience methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectS<T, const D: usize> {
    data: [T; D],
}

impl<T, const D: usize> VectS<T, D> {
    pub const fn new(data: [T; D]) -> Self {
        Self { data: data }
    }

    pub fn cast<S>(&self) -> VectS<S, D>
    where
        T: AsPrimitive<S>,
        S: Copy + 'static,
    {
        VectS::<S, D> {
            data: self.data.map(|x| x.as_()),
        }
    }

    pub fn map<S>(&self, func: impl Fn(T) -> S) -> VectS<S, D>
    where
        T: Copy,
        S: Copy + 'static,
    {
        VectS {
            data: self.data.map(func),
        }
    }

    pub fn map_with_idx<S>(&self, func: impl Fn(usize, T) -> S) -> VectS<S, D>
    where
        T: Copy,
        S: Copy + Default + 'static,
    {
        let mut out = VectS::new([S::default(); D]);
        for i in 0..D {
            out.data[i] = func(i, self.data[i]);
        }
        out
    }

    pub fn sum(&self) -> T
    where
        T: Zero + Add<Output = T> + Copy,
    {
        let mut out = T::zero();
        for d in 0..D {
            out = out + self[d];
        }
        return out;
    }

    pub fn prod(&self) -> T
    where
        T: One + Mul<T> + Copy,
    {
        let mut out = T::one();
        for i in 0..D {
            out = out * self[i];
        }
        out
    }

    pub fn cumprod(&self) -> Self
    where
        T: One + Mul<Output = T> + Copy,
    {
        let mut out = Self::one();
        out[0] = self[0];
        for i in 1..D {
            out[i] = out[i - 1] * self[i];
        }
        out
    }

    pub fn cumsum(&self) -> Self
    where
        T: One + Add<Output = T> + Copy,
    {
        let mut out = Self::one();
        out[0] = self[0];
        for i in 1..D {
            out[i] = out[i - 1] + self[i];
        }
        out
    }
}

impl<T, const D: usize> Into<[T; D]> for VectS<T, D> {
    fn into(self) -> [T; D] {
        return self.data;
    }
}

impl<T, const D: usize> From<[T; D]> for VectS<T, D> {
    fn from(value: [T; D]) -> Self {
        Self::new(value)
    }
}

impl<T, const D: usize> From<PyReadwriteArray1<'_, T>> for VectS<T, D>
where
    T: numpy::Element + Copy + Zero,
{
    fn from(value: PyReadwriteArray1<'_, T>) -> Self {
        assert_eq!(value.len(), D);
        let slice = slice_from_raw_parts(value.data(), value.len());
        let mut data = [T::zero(); D];
        for i in 0..D {
            unsafe { data[i] = *(*slice).get_unchecked(i) }
        }
        Self::new(data)
    }
}

// impl<T, S, const D: usize> Into<VectS<S, D>> for VectS<T, D>
// where
//     T: AsPrimitive<S>,
//     S: Copy,
// {
//     fn into(self) -> VectS<S, D> {
//         return VectS {
//             data: self.data.map(|x| x.as_()),
//         };
//     }
// }

impl<T, const D: usize> Default for VectS<T, D>
where
    T: Zero + Copy,
{
    fn default() -> Self {
        Self {
            data: [T::zero(); D],
        }
    }
}

impl<T, const D: usize> Zero for VectS<T, D>
where
    T: Zero + Copy,
{
    fn zero() -> Self {
        return Self {
            data: [T::zero(); D],
        };
    }

    fn is_zero(&self) -> bool {
        self.data.iter().all(|d| d.is_zero())
    }
}

impl<T, const D: usize> One for VectS<T, D>
where
    T: One + Copy,
{
    fn one() -> Self {
        return Self {
            data: [T::one(); D],
        };
    }
}

impl<T, const D: usize> Index<usize> for VectS<T, D> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        return unsafe { &self.data.get_unchecked(index) };
    }
}

impl<T, const D: usize> IndexMut<usize> for VectS<T, D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        return unsafe { self.data.get_unchecked_mut(index) };
    }
}

impl<T, const D: usize> Add for VectS<T, D>
where
    T: Zero + Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut out = Self::zero();
        for d in 0..D {
            out[d] = self[d] + rhs[d];
        }
        return out;
    }
}

impl<T, const D: usize> Sub for VectS<T, D>
where
    T: Zero + Copy + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = Self::zero();
        for d in 0..D {
            out[d] = self[d] - rhs[d];
        }
        return out;
    }
}

impl<T, const D: usize> Mul for VectS<T, D>
where
    T: One + Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut out = Self::one();
        for d in 0..D {
            out[d] = self[d] * rhs[d];
        }
        return out;
    }
}

impl<T, const D: usize> Div for VectS<T, D>
where
    T: One + Copy + Div<Output = T>,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut out = Self::one();
        for d in 0..D {
            out[d] = self[d] / rhs[d];
        }
        return out;
    }
}

impl<T, const D: usize> Add<T> for VectS<T, D>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Self {
            data: self.data.map(|x| x + rhs),
        }
    }
}

impl<T, const D: usize> Sub<T> for VectS<T, D>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Self {
            data: self.data.map(|x| x - rhs),
        }
    }
}

impl<T, const D: usize> Mul<T> for VectS<T, D>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            data: self.data.map(|x| x * rhs),
        }
    }
}

impl<T, const D: usize> Div<T> for VectS<T, D>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self {
            data: self.data.map(|x| x / rhs),
        }
    }
}

/// impl for concrete f32 rather than T, since rust does not allow the generic one :(
impl<const D: usize> Mul<VectS<f32, D>> for f32 {
    type Output = VectS<f32, D>;

    fn mul(self, rhs: VectS<f32, D>) -> Self::Output {
        rhs * self
    }
}

/// impl for concrete f32 rather than T, since rust does not allow the generic one :(
impl<const D: usize> Div<VectS<f32, D>> for f32 {
    type Output = VectS<f32, D>;

    fn div(self, rhs: VectS<f32, D>) -> Self::Output {
        rhs / self
    }
}

impl<T, const D: usize> AddAssign for VectS<T, D>
where
    T: Copy + AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        for d in 0..D {
            self[d] += rhs[d];
        }
    }
}

impl<T, const D: usize> SubAssign for VectS<T, D>
where
    T: Copy + SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        for d in 0..D {
            self[d] -= rhs[d];
        }
    }
}

impl<T, const D: usize> MulAssign for VectS<T, D>
where
    T: Copy + MulAssign,
{
    fn mul_assign(&mut self, rhs: Self) {
        for d in 0..D {
            self[d] *= rhs[d];
        }
    }
}

impl<T, const D: usize> DivAssign for VectS<T, D>
where
    T: Copy + DivAssign,
{
    fn div_assign(&mut self, rhs: Self) {
        for d in 0..D {
            self[d] /= rhs[d];
        }
    }
}

// Would be nice to implement numpy::Element for VectS so we can use it with the numpy crate,
// but it seems a bunch of the methods needed are private.
// unsafe impl numpy::Element for VectS<f32, 2> {
//     const IS_COPY: bool = true;

//     fn get_dtype(py: pyo3::Python<'_>) -> pyo3::Bound<'_, numpy::PyArrayDescr> {
//         PyArrayDescr::from_npy_type(py, NPY_TYPES::NPY_FLOAT)
//     }

//     fn clone_ref(&self, _py: pyo3::Python<'_>) -> Self {
//         ::std::clone::Clone::clone(self)
//     }
// }
