use num_traits::{One, Zero};

use crate::vect_s::VectS;

#[derive(Debug, Clone, Copy)]
pub struct Sub<const D: usize>(pub VectS<i32, D>);

#[derive(Debug, Clone, Copy)]
pub struct Strides<const D: usize>(pub VectS<i32, D>);

#[derive(Debug, Clone, Copy)]
pub struct Counts<const D: usize>(pub VectS<i32, D>);

/// Ordering to use when calculating strides.
pub enum StrideOrder {
    /// C ordering; last index is contiguous.
    RowMajor,

    /// Fortran ordering; first index is contiguous.
    ColumnMajor,
}

/// Convert domain counts into array strides.
/// See also [`sub_to_idx`].
pub fn counts_to_strides<const D: usize, const ROW_MAJOR: bool>(
    Counts(counts): Counts<D>,
) -> Strides<D> {
    let mut strides = VectS::zero();
    let mut stride = 1;
    for i in 0..D {
        let i = if ROW_MAJOR { D - 1 - i } else { i };
        strides[i] = stride;
        stride *= counts[i];
    }
    return Strides(strides);
}

/// Convert domain counts into array strides.
/// The last index is contiguous.
/// See also [`sub_to_idx`].
pub fn counts_to_strides_row_major<const D: usize>(counts: Counts<D>) -> Strides<D> {
    counts_to_strides::<D, true>(counts)
}

/// Convert domain counts into array strides.
/// The first index is contiguous.
/// See also [`sub_to_idx`].
pub fn counts_to_strides_col_major<const D: usize>(counts: Counts<D>) -> Strides<D> {
    counts_to_strides::<D, false>(counts)
}

pub fn counts_to_strides_dyn<const D: usize>(counts: Counts<D>, order: StrideOrder) -> Strides<D> {
    match order {
        StrideOrder::RowMajor => counts_to_strides_row_major(counts),
        StrideOrder::ColumnMajor => counts_to_strides_col_major(counts),
    }
}

/// Convert a subscript into an index.
/// See also [`counts_to_strides`].
pub fn sub_to_idx<const D: usize>(Sub(sub): Sub<D>, Strides(strides): Strides<D>) -> i32 {
    (strides * sub).sum()
}

/// Move to the next subscript.
pub fn raster_next<const D: usize, const ROW_MAJOR: bool>(
    Sub(mut sub): Sub<D>,
    Counts(counts): Counts<D>,
) -> Sub<D> {
    sub[0] += 1;
    for d in 0..(D - 1) {
        let d = if ROW_MAJOR { D - 1 - d } else { d };
        if sub[d] == counts[d] {
            sub[d] = 0;
            sub[d + 1] += 1;
        } else {
            // If we did not hit the end no need to check others since they won't have been bumped.
            break;
        }
    }
    // if sub[D - 1] == counts[D - 1] {
    //     panic!("Raster past end!")
    // }

    return Sub(sub);
}

/// The first invalid subscript.
pub fn raster_end<const D: usize, const RowMajor: bool>(counts: Counts<D>) -> Sub<D> {
    let end = counts.0 - VectS::one();
    raster_next::<D, RowMajor>(Sub(end), counts)
}

pub fn raster_next_row_major<const D: usize>(sub: Sub<D>, counts: Counts<D>) -> Sub<D> {
    raster_next::<D, true>(sub, counts)
}

pub fn raster_next_col_major<const D: usize>(sub: Sub<D>, counts: Counts<D>) -> Sub<D> {
    raster_next::<D, false>(sub, counts)
}

pub fn raster_end_row_major<const D: usize>(counts: Counts<D>) -> Sub<D> {
    raster_end::<D, true>(counts)
}

pub fn raster_end_col_major<const D: usize>(counts: Counts<D>) -> Sub<D> {
    raster_end::<D, false>(counts)
}

/// Struct which provides convenient iteration over subscripts.
pub struct Raster<const D: usize, const RowMajor: bool> {
    sub: Sub<D>,
    end: Sub<D>,
    cnt: Counts<D>,
}

impl<const D: usize, const RowMajor: bool> Raster<D, RowMajor> {
    pub fn new(counts: Counts<D>) -> Raster<D, RowMajor> {
        let mut sub = VectS::zero();
        sub[0] -= 1;
        Raster {
            sub: Sub(sub),
            end: raster_end::<D, RowMajor>(counts),
            cnt: counts,
        }
    }
}

impl<const D: usize, const RowMajor: bool> Iterator for Raster<D, RowMajor> {
    type Item = Sub<D>;

    fn next(&mut self) -> Option<Self::Item> {
        self.sub = raster_next::<D, RowMajor>(self.sub, self.cnt);
        if self.sub.0 == self.end.0 {
            return None;
        } else {
            return Some(self.sub);
        }
    }
}

/// Iterate over subscripts in row major order.
pub fn raster_row_major<const D: usize>(counts: Counts<D>) -> Raster<D, true> {
    Raster::<D, true>::new(counts)
}

/// Iterate over subscripts in column major order.
pub fn raster_col_major<const D: usize>(counts: Counts<D>) -> Raster<D, false> {
    Raster::<D, false>::new(counts)
}

/// Wrap the compile-time generic [`Raster`] objects in a runtime dynamic enum.
enum RasterDyn<const D: usize> {
    RasterRowMaj(Raster<D, true>),
    RasterColMaj(Raster<D, false>),
}

/// Get a [`Raster`] iterator where you don't know the [`StrideOrder`] until run-time.
pub fn raster_dyn<const D: usize>(counts: Counts<D>, order: StrideOrder) -> RasterDyn<D> {
    match order {
        StrideOrder::RowMajor => RasterDyn::RasterRowMaj(raster_row_major(counts)),
        StrideOrder::ColumnMajor => RasterDyn::RasterColMaj(raster_col_major(counts)),
    }
}

impl<const D: usize> Iterator for RasterDyn<D> {
    type Item = Sub<D>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            RasterDyn::RasterRowMaj(r) => r.next(),
            RasterDyn::RasterColMaj(r) => r.next(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sub_to_idx() {
        let count = Counts(VectS::new([30, 50, 70]));
        let stride = counts_to_strides_col_major(count);
        let sub = Sub(VectS::new([0, 0, 0]));
        assert_eq!(sub_to_idx(sub, stride), 0);

        let sub = Sub(VectS::new([1, 0, 0]));
        assert_eq!(sub_to_idx(sub, stride), 1);

        let sub = Sub(VectS::new([0, 1, 0]));
        assert_eq!(sub_to_idx(sub, stride), 30);

        let sub = Sub(VectS::new([0, 0, 1]));
        assert_eq!(sub_to_idx(sub, stride), 30 * 50);

        let sub = Sub(VectS::new([29, 49, 69]));
        assert_eq!(sub_to_idx(sub, stride), 30 * 50 * 70 - 1);
    }

    #[test]
    fn test_raster() {
        let cnt = Counts(VectS::new([2, 2, 2]));
        let mut sub = Sub(VectS::new([0, 0, 0]));

        sub = raster_next_col_major(sub, cnt);
        assert_eq!(sub.0, VectS::new([1, 0, 0]));

        sub = raster_next_col_major(sub, cnt);
        assert_eq!(sub.0, VectS::new([0, 1, 0]));

        sub = raster_next_col_major(sub, cnt);
        assert_eq!(sub.0, VectS::new([1, 1, 0]));

        sub = raster_next_col_major(sub, cnt);
        assert_eq!(sub.0, VectS::new([0, 0, 1]));

        sub = raster_next_col_major(sub, cnt);
        assert_eq!(sub.0, VectS::new([1, 0, 1]));

        sub = raster_next_col_major(sub, cnt);
        assert_eq!(sub.0, VectS::new([0, 1, 1]));

        sub = raster_next_col_major(sub, cnt);
        assert_eq!(sub.0, VectS::new([1, 1, 1]));

        // TODO: check one more panics
        // sub = raster(sub, cnt);
    }
}
