use super::from_element::MatrixFill;

pub fn fill_eye<T: MatrixFill>(n: usize) -> Vec<T> {
    let mut v = vec![T::zero_val(); n * n];
    for i in 0..n {
        v[i * n + i] = T::one_val();
    }
    v
}

pub fn fill_tril<T: MatrixFill>(n: usize, diagonal: bool) -> Vec<T> {
    let mut v = vec![T::one_val(); n * n];
    for i in 0..n {
        let end = if diagonal { i + 1 } else { i };
        for j in end..n {
            v[i * n + j] = T::zero_val();
        }
    }
    v
}

pub fn fill_triu<T: MatrixFill>(n: usize, diagonal: bool) -> Vec<T> {
    let mut v = vec![T::one_val(); n * n];
    for i in 0..n {
        let start = if diagonal { i } else { i + 1 };
        for j in 0..start {
            v[i * n + j] = T::zero_val();
        }
    }
    v
}
