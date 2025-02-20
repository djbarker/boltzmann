use ndarray::Array1;

use crate::raster::StrideOrder::{ColumnMajor, RowMajor};

/// We sometimes need subscript values to be negative.
pub type Ix = i32;

/// Things that can be converted into an [`Ix`].
pub trait IxLike: Into<Ix> + Copy {}

impl IxLike for i16 {}
impl IxLike for i32 {}

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

    fn reverse(&self) -> StrideOrder {
        match self {
            RowMajor => ColumnMajor,
            ColumnMajor => RowMajor,
        }
    }
}

/// Convert domain counts into array strides along each dimension.
///
/// See also [`sub_to_idx`].
pub fn counts_to_strides(counts: &Array1<Ix>, order: StrideOrder) -> Array1<Ix> {
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
pub fn sub_to_idx(sub: &Array1<Ix>, strides: &Array1<Ix>) -> usize {
    (strides * sub).sum() as usize
}

/// Convert an index into a subscript.
pub fn idx_to_sub(idx: usize, strides: &Array1<Ix>, order: StrideOrder) -> Array1<Ix> {
    let ndim = strides.dim();
    let mut idx = idx as Ix;
    let mut sub = Array1::zeros([ndim]);
    for d in 0..ndim {
        let d = order.reverse().sub_idx(ndim, d);
        sub[d] = idx / strides[d];
        idx = idx % strides[d];
    }
    sub
}

/// Move to the next subscript.
pub fn raster_next(mut sub: Array1<Ix>, counts: &Array1<Ix>, order: StrideOrder) -> Array1<Ix> {
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
pub fn raster_end(counts: &Array1<Ix>, order: StrideOrder) -> Array1<Ix> {
    let ones = Array1::<Ix>::ones([counts.dim()]);
    let end = counts - ones;
    raster_next(end, counts, order)
}

/// Struct which provides convenient iteration over subscripts.
///
/// ## Example
/// ```
/// for sub in Raster::new(arr1([5, 7]), StrideOrder::RowMajor) {
///     println!("{:?}", sub);
/// }
/// ```
pub struct Raster {
    sub: Array1<Ix>,
    end: Array1<Ix>,
    cnt: Array1<Ix>,
    ord: StrideOrder,
}

impl Raster {
    pub fn dim(&self) -> usize {
        self.cnt.dim()
    }

    pub fn new<T: Into<Ix> + Copy>(counts: Array1<T>, order: StrideOrder) -> Self {
        let counts = counts.mapv(|c| c.into());
        let ndim = counts.dim();
        let shape = [ndim];
        let mut sub = Array1::zeros(shape);
        let d = order.sub_idx(ndim, 0);
        sub[d] -= 1; // start from one before the beginning
        Raster {
            sub: sub,
            end: raster_end(&counts, order),
            cnt: counts,
            ord: order,
        }
    }
}

impl Iterator for Raster {
    type Item = Array1<Ix>;

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
///
/// ## Example
/// ```
/// for sub in raster_row_major(arr1([5, 7])) {
///     println!("{:?}", sub);
/// }
pub fn raster_row_major<T: Into<Ix> + Copy>(counts: Array1<T>) -> Raster {
    Raster::new(counts, RowMajor)
}

/// Convenience function to iterate over subscripts in [`ColumnMajor`] order.
///
/// ## Example
/// ```
/// for sub in raster_col_major(arr1([5, 7])) {
///     println!("{:?}", sub);
/// }
pub fn raster_col_major<T: Into<Ix> + Copy>(counts: Array1<T>) -> Raster {
    Raster::new(counts, ColumnMajor)
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*;

    #[test]
    fn test_sub_to_idx_col_major() {
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
    fn test_sub_to_idx_row_major() {
        let count = arr1(&[30, 50, 70]);
        let stride = counts_to_strides(&count, RowMajor);
        let sub = arr1(&[0, 0, 0]);
        assert_eq!(sub_to_idx(&sub, &stride), 0);

        let sub = arr1(&[0, 0, 1]);
        assert_eq!(sub_to_idx(&sub, &stride), 1);

        let sub = arr1(&[0, 1, 0]);
        assert_eq!(sub_to_idx(&sub, &stride), 70);

        let sub = arr1(&[1, 0, 0]);
        assert_eq!(sub_to_idx(&sub, &stride), 50 * 70);

        let sub = arr1(&[29, 49, 69]);
        assert_eq!(sub_to_idx(&sub, &stride), 30 * 50 * 70 - 1);
    }

    #[test]
    fn test_raster_col_major() {
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

    #[test]
    fn test_raster_row_major() {
        let cnt = arr1(&[2, 2, 2]);
        let mut sub = arr1(&[0, 0, 0]);

        sub = raster_next(sub, &cnt, RowMajor);
        assert_eq!(sub, arr1(&[0, 0, 1]));

        sub = raster_next(sub, &cnt, RowMajor);
        assert_eq!(sub, arr1(&[0, 1, 0]));

        sub = raster_next(sub, &cnt, RowMajor);
        assert_eq!(sub, arr1(&[0, 1, 1]));

        sub = raster_next(sub, &cnt, RowMajor);
        assert_eq!(sub, arr1(&[1, 0, 0]));

        sub = raster_next(sub, &cnt, RowMajor);
        assert_eq!(sub, arr1(&[1, 0, 1]));

        sub = raster_next(sub, &cnt, RowMajor);
        assert_eq!(sub, arr1(&[1, 1, 0]));

        sub = raster_next(sub, &cnt, RowMajor);
        assert_eq!(sub, arr1(&[1, 1, 1]));

        sub = raster_next(sub, &cnt, RowMajor);
        assert_eq!(sub, raster_end(&cnt, RowMajor))
    }

    #[test]
    fn test_idx_to_sub_col_major() {
        let cnt = arr1(&[7, 5, 3]);
        let strides = counts_to_strides(&cnt, ColumnMajor);
        for sub1 in raster_col_major(cnt) {
            let idx = sub_to_idx(&sub1, &strides);
            let sub2 = idx_to_sub(idx, &strides, ColumnMajor);
            assert_eq!(sub1, sub2);
        }
    }

    #[test]
    fn test_idx_to_sub_row_major() {
        let cnt = arr1(&[7, 5, 3]);
        let strides = counts_to_strides(&cnt, RowMajor);
        for sub1 in raster_row_major(cnt) {
            let idx = sub_to_idx(&sub1, &strides);
            let sub2 = idx_to_sub(idx, &strides, RowMajor);
            assert_eq!(sub1, sub2);
        }
    }
}
