use std::{
    ops::{Index, IndexMut},
    ptr::{slice_from_raw_parts, slice_from_raw_parts_mut},
};

use num_traits::{One, Zero};
use numpy::{PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods};

use crate::vect_s::VectS;

/// Allows access to the underlying primative type.
/// When storing a primative this is just `Self`, but for [`VectS`] it will return the element type.
pub trait Data {
    const STRIDE: usize;
    type Elem;
}

impl Data for f32 {
    const STRIDE: usize = 1;
    type Elem = Self;
}

impl Data for i32 {
    const STRIDE: usize = 1;
    type Elem = Self;
}

impl Data for bool {
    const STRIDE: usize = 1;
    type Elem = Self;
}

impl<T, const D: usize> Data for VectS<T, D> {
    const STRIDE: usize = D;
    type Elem = T;
}

/// A trait for arrays (vectors) we can index into with i32 without bounds checking, and easily
/// convert to various ptr and slice types of the undlerying type for interacting with Python
/// (see [`Data`].)
///
/// The underlying type will either be a [`VectD`] or a [`VectDView`].
/// [`VectD`] owns its data, wheras [`VectDView`] points to data owned by somebody else.
/// Both are mutable.
///
/// Having this trait with its two implementations means our library functions can be agnostic to
/// the ownership of the array by taking `impl [ArrayD]` as the argument type.
/// With `ndarray` I think the way to do it would be to make the function explicitly generic over
/// the array type.
/// Actually, I think it's a bit clunky how it currently is. The view is effectively nothing
/// more than a fat pointer so isn't too heavy to just copy as pass as an argument.
/// Therefore perhaps they should all just take the view as arguments and we can even do away
/// with the genericness.
///
/// TODO: I'm not happy about the naming of these three things.
pub trait ArrayD<T>: Index<i32, Output = T> + IndexMut<i32, Output = T>
where
    T: Data,
{
    fn data_count(&self) -> i32;
    fn elem_count(&self) -> i32 {
        self.data_count() * (T::STRIDE as i32)
    }
    fn as_ptr(&self) -> *const T::Elem;
    fn as_slice_ptr(&self) -> *const [T::Elem];
    fn as_slice(&self) -> &[T::Elem];
    fn as_mut_ptr(&mut self) -> *mut T::Elem;
    fn as_mut_slice_ptr(&mut self) -> *mut [T::Elem];
    fn as_mut_slice(&mut self) -> &mut [T::Elem];
}

#[derive(Debug, Clone, Default)]
pub struct VectD<T> {
    pub data: Vec<T>,
}

impl<T> VectD<T> {
    pub fn zeros(size: usize) -> VectD<T>
    where
        T: Zero + Copy,
    {
        Self {
            data: vec![T::zero(); size],
        }
    }

    pub fn ones(size: usize) -> VectD<T>
    where
        T: One + Copy,
    {
        Self {
            data: vec![T::one(); size],
        }
    }

    pub fn map<S>(self, func: fn(T) -> S) -> VectD<S> {
        VectD {
            data: self.data.into_iter().map(func).collect(),
        }
    }
}

impl<T, const N: usize> From<[T; N]> for VectD<T> {
    fn from(value: [T; N]) -> Self {
        Self { data: value.into() }
    }
}

impl<T> ArrayD<T> for VectD<T>
where
    T: Data,
{
    fn data_count(&self) -> i32 {
        self.data.len() as i32
    }

    fn as_ptr(&self) -> *const T::Elem {
        self.data.as_ptr() as *const T::Elem
    }

    fn as_slice_ptr(&self) -> *const [T::Elem] {
        slice_from_raw_parts(self.as_ptr(), self.elem_count() as usize)
    }

    fn as_slice(&self) -> &[T::Elem] {
        unsafe { &*self.as_slice_ptr() }
    }

    fn as_mut_ptr(&mut self) -> *mut T::Elem {
        self.data.as_mut_ptr() as *mut T::Elem
    }

    fn as_mut_slice_ptr(&mut self) -> *mut [T::Elem] {
        slice_from_raw_parts_mut(self.as_mut_ptr(), self.elem_count() as usize)
    }

    fn as_mut_slice(&mut self) -> &mut [T::Elem] {
        unsafe { &mut *self.as_mut_slice_ptr() }
    }
}

impl<T> Index<i32> for VectD<T> {
    type Output = T;

    fn index(&self, index: i32) -> &Self::Output {
        unsafe { &self.data.get_unchecked(index as usize) }
    }
}

impl<T> IndexMut<i32> for VectD<T> {
    fn index_mut(&mut self, index: i32) -> &mut Self::Output {
        unsafe { self.data.get_unchecked_mut(index as usize) }
    }
}

pub struct VectDView<T>
where
    T: Data,
{
    pub data: *mut [T],
}

impl<T> ArrayD<T> for VectDView<T>
where
    T: Data,
{
    fn data_count(&self) -> i32 {
        self.data.len() as i32
    }

    fn as_ptr(&self) -> *const T::Elem {
        unsafe { (*self.data).as_ptr() as *const T::Elem }
    }

    fn as_slice_ptr(&self) -> *const [T::Elem] {
        slice_from_raw_parts(self.as_ptr(), self.elem_count() as usize)
    }

    fn as_slice(&self) -> &[T::Elem] {
        unsafe { &*self.as_slice_ptr() }
    }

    fn as_mut_ptr(&mut self) -> *mut T::Elem {
        unsafe { (*self.data).as_mut_ptr() as *mut T::Elem }
    }

    fn as_mut_slice_ptr(&mut self) -> *mut [T::Elem] {
        slice_from_raw_parts_mut(self.as_mut_ptr(), self.elem_count() as usize)
    }

    fn as_mut_slice(&mut self) -> &mut [T::Elem] {
        // unsafe { &mut (*(self.data as *mut [T::Elem])) }
        unsafe { &mut *self.as_mut_slice_ptr() }
    }
}

impl<T> Index<i32> for VectDView<T>
where
    T: Data,
{
    type Output = T;

    fn index(&self, index: i32) -> &Self::Output {
        unsafe { &(*self.data).get_unchecked(index as usize) }
    }
}

impl<T> IndexMut<i32> for VectDView<T>
where
    T: Data,
{
    fn index_mut(&mut self, index: i32) -> &mut Self::Output {
        unsafe { (*self.data).get_unchecked_mut(index as usize) }
    }
}

impl<T> From<PyReadwriteArray1<'_, T::Elem>> for VectDView<T>
where
    T: Data,
    T::Elem: numpy::Element,
{
    fn from(mut arr: PyReadwriteArray1<T::Elem>) -> Self {
        assert_eq!(arr.len() % T::STRIDE, 0); // Whole number of Ts please.
        let n = arr.len() / T::STRIDE; // Number of T objects.
        let p = arr.as_array_mut().as_mut_ptr();
        Self {
            data: slice_from_raw_parts_mut(p as *mut T, n),
        }
    }
}

impl<T, const D: usize> From<PyReadwriteArray2<'_, T>> for VectDView<VectS<T, D>>
where
    T: numpy::Element,
{
    fn from(mut arr: PyReadwriteArray2<'_, T>) -> Self {
        assert_eq!(D * size_of::<T>(), size_of::<VectS<T, D>>()); // Correct sizes please.
        assert_eq!(arr.len() % D, 0); // Whole number of VectS please.
        let n = arr.len() / D; // Number of VectS objects.
        let p = arr.as_array_mut().as_mut_ptr();
        let p = p as *mut VectS<T, D>;
        Self {
            data: slice_from_raw_parts_mut(p, n),
        }
    }
}

pub type VectDViewScalar<T> = VectDView<T>;
pub type VectDViewVector<T, const D: usize> = VectDView<VectS<T, D>>;
