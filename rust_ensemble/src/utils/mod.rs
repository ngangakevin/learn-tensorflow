use burn::tensor::backend::{Backend};
use burn::prelude::TensorData;
use burn::tensor::{Tensor, Shape};
use safetensors::SafeTensors;

pub fn load_tensor<B: Backend>(
    name: &str,
    tensors: &SafeTensors,
    device: &B::Device,
) -> Tensor<B, 4> {
    let view = tensors.tensor(name).unwrap_or_else(|_| panic!("Tensor '{}' not found", name));

    let f32_data: &[f32] = bytemuck::cast_slice(view.data());
    let shape = Shape::from(view.shape().to_vec());
    let tensor = Tensor::<B, 4>::from_data(TensorData::new(f32_data.to_vec(), shape), device);
    tensor.permute([3,2,0,1])
}

pub fn load_flat_tensor<B: Backend, const D: usize>(
    name: &str,
    tensors: &SafeTensors,
    device: &B::Device,
)-> Tensor<B, D> {
    let view =  tensors.tensor(name).unwrap_or_else(|_| panic!(
        "Tensor '{}' not found", name));
    let f32_data: &[f32] = bytemuck::cast_slice(view.data());
    let shape = Shape::from(view.shape().to_vec());

    Tensor::<B, D>::from_data(TensorData::new(f32_data.to_vec(), shape), device)
}
