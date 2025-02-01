use ndarray::Array1;

use crate::raster::StrideOrder::{ColumnMajor, RowMajor};
use crate::vect_s::VectS;

#[derive(Debug, Clone, Copy)]
pub struct Sub<const D: usize>(pub VectS<i32, D>);

#[derive(Debug, Clone, Copy)]
pub struct Strides<const D: usize>(pub VectS<i32, D>);

#[derive(Debug, Clone, Copy)]
pub struct Counts<const D: usize>(pub VectS<i32, D>);

/// Ordering to use when calculating strides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrideOrder {
    /// C ordering; last index is contiguous.
    RowMajor,

    /// Fortran ordering; first index is contiguous.
    ColumnMajor,
}

impl StrideOrder {
    fn sub_idx(&self, ndim: usize, d: usize) -> usize {
        match self {
            RowMajor => ndim - 1 - d,
            ColumnMajor => d,
        }
    }

    fn next_idx(&self, d: usize) -> usize {
        match self {
            RowMajor => ((d as isize) - 1) as usize,
            ColumnMajor => d + 1,
        }
    }
}

/// Convert domain counts into array strides along each dimension.
///
/// See also [`sub_to_idx`].
pub fn counts_to_strides(counts: &Array1<i32>, order: StrideOrder) -> Array1<i32> {
    let mut strides = Array1::zeros(counts.dim());
    let mut stride = 1;
    for i in 0..counts.dim() {
        let i = order.sub_idx(counts.dim(), i);
        strides[i] = stride;
        stride *= counts[i];
    }
    strides
}

/// Convert a subscript into an index.
///
/// See also [`counts_to_strides`] for the second argument.
pub fn sub_to_idx(sub: &Array1<i32>, strides: &Array1<i32>) -> i32 {
    (strides * sub).sum()
}

/// Move to the next subscript.
pub fn raster_next(mut sub: Array1<i32>, counts: &Array1<i32>, order: StrideOrder) -> Array1<i32> {
    assert_eq!(counts.dim(), sub.dim());

    let ndim = counts.dim();
    let d = order.sub_idx(ndim, 0);
    sub[d] += 1;
    for d in 0..(ndim - 1) {
        let d = order.sub_idx(ndim, d);
        if sub[d] == counts[d] {
            sub[d] = 0;
            sub[order.next_idx(d)] += 1;
        } else {
            // If we did not hit the end no need to check others since they won't have been bumped.
            break;
        }
    }

    sub
}

/// The first invalid subscript.
pub fn raster_end(counts: &Array1<i32>, order: StrideOrder) -> Array1<i32> {
    let ones = Array1::<i32>::ones([counts.dim()]);
    let end = counts - ones;
    raster_next(end, counts, order)
}

/// Struct which provides convenient iteration over subscripts.
pub struct Raster {
    sub: Array1<i32>,
    end: Array1<i32>,
    cnt: Array1<i32>,
    ord: StrideOrder,
}

impl Raster {
    pub fn dim(&self) -> usize {
        self.cnt.dim()
    }

    pub fn new(counts: Array1<i32>, order: StrideOrder) -> Self {
        let ndim = counts.dim();
        let shape = [ndim];
        let mut sub = Array1::zeros(shape);
        let d = order.sub_idx(ndim, 0);
        sub[d] -= 1; // start from one past the beginning
        Raster {
            sub: sub,
            end: raster_end(&counts, order),
            cnt: counts,
            ord: order,
        }
    }
}

impl Iterator for Raster {
    type Item = Array1<i32>;

    fn next(&mut self) -> Option<Self::Item> {
        self.sub = raster_next(self.sub.clone(), &self.cnt, self.ord);
        if self.sub == self.end {
            return None;
        } else {
            return Some(self.sub.clone());
        }
    }
}

/// Convenience function to iterate over subscripts in [`RowMajor`] order.
pub fn raster_row_major(counts: Array1<i32>) -> Raster {
    Raster::new(counts, RowMajor)
}

/// Convenience function to iterate over subscripts in [`ColumnMajor`] order.
pub fn raster_col_major(counts: Array1<i32>) -> Raster {
    Raster::new(counts, ColumnMajor)
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*;

    #[test]
    fn test_sub_to_idx() {
        let count = arr1(&[30, 50, 70]);
        let stride = counts_to_strides(&count, ColumnMajor);
        let sub = arr1(&[0, 0, 0]);
        assert_eq!(sub_to_idx(&sub, &stride), 0);

        let sub = arr1(&[1, 0, 0]);
        assert_eq!(sub_to_idx(&sub, &stride), 1);

        let sub = arr1(&[0, 1, 0]);
        assert_eq!(sub_to_idx(&sub, &stride), 30);

        let sub = arr1(&[0, 0, 1]);
        assert_eq!(sub_to_idx(&sub, &stride), 30 * 50);

        let sub = arr1(&[29, 49, 69]);
        assert_eq!(sub_to_idx(&sub, &stride), 30 * 50 * 70 - 1);
    }

    #[test]
    fn test_raster() {
        let cnt = arr1(&[2, 2, 2]);
        let mut sub = arr1(&[0, 0, 0]);

        sub = raster_next(sub, &cnt, ColumnMajor);
        assert_eq!(sub, arr1(&[1, 0, 0]));

        sub = raster_next(sub, &cnt, ColumnMajor);
        assert_eq!(sub, arr1(&[0, 1, 0]));

        sub = raster_next(sub, &cnt, ColumnMajor);
        assert_eq!(sub, arr1(&[1, 1, 0]));

        sub = raster_next(sub, &cnt, ColumnMajor);
        assert_eq!(sub, arr1(&[0, 0, 1]));

        sub = raster_next(sub, &cnt, ColumnMajor);
        assert_eq!(sub, arr1(&[1, 0, 1]));

        sub = raster_next(sub, &cnt, ColumnMajor);
        assert_eq!(sub, arr1(&[0, 1, 1]));

        sub = raster_next(sub, &cnt, ColumnMajor);
        assert_eq!(sub, arr1(&[1, 1, 1]));

        sub = raster_next(sub, &cnt, ColumnMajor);
        assert_eq!(sub, raster_end(&cnt, ColumnMajor))
    }
}
