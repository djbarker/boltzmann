/// Calculate the domain decomposition counts in each dimension.
///
/// The decomposition is always such that subdomain cuboids have their verticies touching,
/// edges touching, etc.
/// It tries to keep the subdomains as close to cubic as possible given this constraint.
pub fn decomp_2d(ncpus: usize) -> [usize; 2] {
    let mut root = 1;
    for i in 2..=(ncpus as f32).sqrt() as usize {
        if ncpus % i == 0 {
            root = i;
        }
    }

    [root, ncpus / root]
}

/// Calculate the domain decomposition counts in each dimension.
///
/// The decomposition is always such that subdomain cuboids have their vertices touching,
/// edges touching, etc.
/// It tries to keep the subdomains as close to cubic as possible given this constraint.
pub fn decomp_3d(ncpus: usize) -> [usize; 3] {
    let mut root = 1;
    let imax = (ncpus as f32).powf(1.0 / 3.0) as usize;
    for i in 2..=imax {
        if ncpus % i == 0 {
            root = i;
        }
    }

    let [ny, nx] = decomp_2d(ncpus / root);
    [root, ny, nx]
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
        assert_eq!(decomp_3d(1), [1, 1, 1]);
        assert_eq!(decomp_3d(2), [1, 1, 2]);
        assert_eq!(decomp_3d(3), [1, 1, 3]);
        assert_eq!(decomp_3d(4), [1, 2, 2]);
        assert_eq!(decomp_3d(5), [1, 1, 5]);
        assert_eq!(decomp_3d(6), [1, 2, 3]);
        assert_eq!(decomp_3d(7), [1, 1, 7]);
        assert_eq!(decomp_3d(8), [2, 2, 2]);
        assert_eq!(decomp_3d(9), [1, 3, 3]);
        assert_eq!(decomp_3d(10), [1, 2, 5]);
    }
}
