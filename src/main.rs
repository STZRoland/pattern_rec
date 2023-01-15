use pattern_rec::dim_reduction::pca;
use ndarray::{Array, Ix2};

fn main() {
    let mut test = Array::<f32, Ix2>::ones((15, 128));

    test[[0, 0]] = 0.0_f32;


    let pca = pca::PCA::new(2);
    pca.fit(&test);
    let transformed = pca.transform(&test);
}