use std::{
    ops::{Index, IndexMut},
    ptr::slice_from_raw_parts_mut,
    str,
};

use num_traits::{One, Zero};
use numpy::{Element, PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods};

use crate::vect_s::VectS;

/// A trait for arrays (vectors) we can index into with i32 without bounds checking.
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
pub trait ArrayD<T>: Index<i32, Output = T> + IndexMut<i32, Output = T> {
    fn len(&self) -> i32;
}

#[derive(Debug, Clone)]
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

impl<T> ArrayD<T> for VectD<T> {
    fn len(&self) -> i32 {
        self.data.len() as i32
    }
}

impl<T> Index<i32> for VectD<T> {
    type Output = T;

    fn index(&self, index: i32) -> &Self::Output {
        return unsafe { &self.data.get_unchecked(index as usize) };
    }
}

impl<T> IndexMut<i32> for VectD<T> {
    fn index_mut(&mut self, index: i32) -> &mut Self::Output {
        return unsafe { self.data.get_unchecked_mut(index as usize) };
    }
}

pub struct VectDView<T> {
    pub data: *mut [T],
    pub size: i32,
}

impl<T> ArrayD<T> for VectDView<T> {
    fn len(&self) -> i32 {
        self.size
    }
}

impl<T> Index<i32> for VectDView<T> {
    type Output = T;

    fn index(&self, index: i32) -> &Self::Output {
        return unsafe { &(*self.data).get_unchecked(index as usize) };
    }
}

impl<T> IndexMut<i32> for VectDView<T> {
    fn index_mut(&mut self, index: i32) -> &mut Self::Output {
        return unsafe { (*self.data).get_unchecked_mut(index as usize) };
    }
}

impl<T> From<PyReadwriteArray1<'_, T>> for VectDView<T>
where
    T: numpy::Element,
{
    fn from(mut arr: PyReadwriteArray1<T>) -> Self {
        let n = arr.len();
        let p = arr.as_array_mut().as_mut_ptr();
        Self {
            data: slice_from_raw_parts_mut(p, n),
            size: n as i32,
        }
    }
}

impl<T, const D: usize> From<PyReadwriteArray2<'_, T>> for VectDView<VectS<T, D>>
where
    T: numpy::Element,
{
    fn from(mut arr: PyReadwriteArray2<'_, T>) -> Self {
        assert_eq!(D * size_of::<T>(), size_of::<VectS<T, D>>());
        let n = arr.len() / D; // number of VectS objects
        let p = arr.as_array_mut().as_mut_ptr();
        let p = p as *mut VectS<T, D>;
        Self {
            data: slice_from_raw_parts_mut(p, n),
            size: n as i32,
        }
    }
}
