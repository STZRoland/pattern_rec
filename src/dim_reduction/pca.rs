use ndarray::{Array, Ix2, Ix1, Axis, s, concatenate};
use ndarray_linalg::Eig;
use ndarray_stats::CorrelationExt;

pub struct PCA {
    pub n_components: u32,
    // eigenvectors: ArrayBase<OwnedRepr<f32>, Ix2>
    principal_components: Array<f32, Ix2>,
    lambda: Array<f32, Ix1>,
}


impl PCA {

    pub fn new(n_components: u32) -> Self {
        return Self {
            n_components: n_components,
            principal_components: Array::<f32, Ix2>::default((10, n_components as usize)),
            lambda: Array::<f32, Ix1>::default(n_components as usize)
        };
    }

    pub fn fit(&mut self, data: &Array<f32, Ix2>) {
        let cov = data.cov(0.0).unwrap();
        // println!("{:?}", cov);
        
        let (eigenvalues, eigenvectors) = cov.eig().unwrap();

        // Closure function: |eigenvectors| eigenvectors.re equivalent to def real_part(eigenvectors): return eigenvectors.re

        // self.principal_components = eigenvectors.mapv(|vec| vec.re).slice(s![..self.n_components, ..]);
        // self.lambda = eigenvalues.mapv(|vec| vec.re).slice(s![..self.n_components]);

        self.principal_components = Array::<f32, Ix2>::default((data.shape()[0], self.n_components as usize));
        self.principal_components = eigenvectors.mapv(|vec| vec.re);
        self.lambda = eigenvalues.mapv(|vec| vec.re);

    }

    pub fn transform(&self, data: &Array<f32, Ix2>) -> Array<f32, Ix2> {
        
        // let mut transformed_data = data.map_axis(Axis(0), |row| {
        //     let mut transformed_row = Array::<f32, Ix1>::default(self.n_components as usize);
        // let mut transformed_data = Array::<f32, Ix2>::default((data.shape()[0], self.n_components as usize));            
        // for (i, mut row) in transformed_data.axis_iter_mut(Axis(0)).enumerate() {
        //     for (j, value) in row.indexed_iter_mut() {
        //         *value = row.dot(&self.principal_components.slice(s![i , ..]));
        //     }
        // };

        let mut transformed_data = Array::<f32, Ix2>::default((data.shape()[0], self.n_components as usize));
        for (i, mut row) in transformed_data.axis_iter_mut(Axis(0)).enumerate() {
            for (j, value) in row.indexed_iter_mut() {
                *value = data.slice(s![.., i]).dot(&self.principal_components.slice(s![j, ..]));
            }
        }
        return transformed_data;
    }
}