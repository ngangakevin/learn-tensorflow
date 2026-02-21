use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, Shape, Param};
use safetensors::SafeTensors;
pub fn load_tensor<B: Backend, const D: usize>(
    name: &str,
    tensors: &SafeTensors,
    device: &Device,
) -> Tensor<B, D> {
    let view = tensors.tensor(name).unwrap_or_else(|_| panic!("Tensor '{}' not found", name));

    let f32_data: &[f32] = bytemuck::cast_slice(view.data());
    let shape = Shape::from(view.shape().to_vec());
    Tensor::<B, D>::from_data(f32_data, &device).reshape(shape)
}

pub fn load_model_weights<B: Backend>(
    mode: &mut LeNet5BNN<B>, tensors: &SafeTensors, device: &B::Device) {
    model.conv1.kernel_posterior_loc = Param::from_tensor(
        load_tensor("conv1.kernel_posterior_loc", tensors, device));
    model.conv1.kernel_posterior_untransformed_scale = Param::from_tensor(
        load_tensor("conv1.kernel_posterior_untransformed_scale", tensors, device));
    model.conv1.bias_posterior_loc = Param::from_tensor(
        load_tensor("conv1.bias_posterior_loc", tensors, device));
}