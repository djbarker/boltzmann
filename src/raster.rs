use num_traits::{One, Zero};

use crate::vect_s::VectS;

/// Convert a subscript into an index.
/// The zeroth dimension is contiguous.
pub fn sub_to_idx<const D: usize>(sub: VectS<i32, D>, counts: VectS<i32, D>) -> i32 {
    let mut idx = 0;
    let mut stride = 1;
    for i in 0..D {
        idx += stride * sub[i];
        stride *= counts[i] as i32;
    }
    idx
}

/// Move to the next subscript.
pub fn raster<const D: usize>(mut sub: VectS<i32, D>, counts: VectS<i32, D>) -> VectS<i32, D> {
    sub[0] += 1;
    for d in 0..(D - 1) {
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
    sub
}

/// The first invalid subscript.
pub fn raster_end<const D: usize>(counts: VectS<i32, D>) -> VectS<i32, D> {
    let end = counts - VectS::one();
    raster(end, counts)
}

/// Struct which provides convenient iteration over subscripts.
pub struct Raster<const D: usize> {
    sub: VectS<i32, D>,
    cnt: VectS<i32, D>,
    end: VectS<i32, D>,
}

impl<const D: usize> Raster<D> {
    pub fn new(cnt: VectS<i32, D>) -> Raster<D> {
        let mut sub = VectS::zero();
        sub[0] -= 1;
        Raster {
            sub: sub,
            cnt: cnt,
            end: raster_end(cnt),
        }
    }
}

impl<const D: usize> Iterator for Raster<D> {
    type Item = VectS<i32, D>;

    fn next(&mut self) -> Option<Self::Item> {
        self.sub = raster(self.sub, self.cnt);
        if self.sub == self.end {
            return None;
        } else {
            return Some(self.sub);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sub_to_idx() {
        let cnt = VectS::new([30, 50, 70]);
        let sub = VectS::new([0, 0, 0]);
        assert_eq!(sub_to_idx(sub, cnt), 0);

        let sub = VectS::new([1, 0, 0]);
        assert_eq!(sub_to_idx(sub, cnt), 1);

        let sub = VectS::new([0, 1, 0]);
        assert_eq!(sub_to_idx(sub, cnt), 30);

        let sub = VectS::new([0, 0, 1]);
        assert_eq!(sub_to_idx(sub, cnt), 30 * 50);

        let sub = VectS::new([29, 49, 69]);
        assert_eq!(sub_to_idx(sub, cnt), 30 * 50 * 70 - 1);
    }

    #[test]
    fn test_raster() {
        let cnt = VectS::new([2, 2, 2]);
        let mut sub = VectS::new([0, 0, 0]);

        sub = raster(sub, cnt);
        assert_eq!(sub, VectS::new([1, 0, 0]));

        sub = raster(sub, cnt);
        assert_eq!(sub, VectS::new([0, 1, 0]));

        sub = raster(sub, cnt);
        assert_eq!(sub, VectS::new([1, 1, 0]));

        sub = raster(sub, cnt);
        assert_eq!(sub, VectS::new([0, 0, 1]));

        sub = raster(sub, cnt);
        assert_eq!(sub, VectS::new([1, 0, 1]));

        sub = raster(sub, cnt);
        assert_eq!(sub, VectS::new([0, 1, 1]));

        sub = raster(sub, cnt);
        assert_eq!(sub, VectS::new([1, 1, 1]));

        // TODO: check one more panics
        // sub = raster(sub, cnt);
    }
}
