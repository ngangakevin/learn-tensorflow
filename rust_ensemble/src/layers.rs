use burn::config::Config;
use burn::module::{Module, Param};
use burn::tensor::{backend::Backend, Tensor, Distribution};
use burn::tensor::activation::softplus;
use burn::tensor::ops::ConvOptions;

#[derive(Config, Debug)]
pub struct DenseFlipoutConfig {
    pub input_size: usize,
    pub output_size: usize,
}

#[derive(Module, Debug)]
pub struct DenseFlipout<B: Backend> {
    pub kernel_posterior_loc: Param<Tensor<B, 2>>,
    pub kernel_posterior_untransformed_scale: Param<Tensor<B, 2>>,
    pub bias_posterior_loc: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
pub struct Conv2dFlipout<B: Backend> {
    pub kernel_posterior_loc: Param<Tensor<B, 4>>,
    pub kernel_posterior_untransformed_scale: Param<Tensor<B, 4>>,
    pub bias_posterior_loc: Param<Tensor<B, 1>>,
    pub stride: [usize; 2],
    pub padding: [usize; 2],
}
#[derive(Config, Debug)]
pub struct Conv2dFlipoutConfig {
    pub channels: [usize; 2],
    pub kernel_size: [usize; 2],
    #[config(default="[1, 1]")]
    pub stride: [usize; 2],
    #[config(default="[2, 2]")]
    pub padding: [usize; 2],
}

impl Conv2dFlipoutConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Conv2dFlipout<B> {
        let shape = [self.channels[1], self.channels[0], self.kernel_size[0], self.kernel_size[1]];
        Conv2dFlipout {
            kernel_posterior_loc: Param::from_tensor(Tensor::zeros(shape.clone(), device)),
            kernel_posterior_untransformed_scale: Param::from_tensor(Tensor::zeros(shape, device)),
            bias_posterior_loc: Param::from_tensor(Tensor::zeros([self.channels[1]], device)),
            stride: self.stride,
            padding: self.padding,
        }
    }
}

impl<B: Backend> Conv2dFlipout<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let device = self.kernel_posterior_loc.device();
        let shape = self.kernel_posterior_loc.shape();

        let sigma = softplus(self.kernel_posterior_untransformed_scale.val(), 1.0 );

        let dist = Distribution::Normal(0.0, 1.0);
        let epsilon = Tensor::<B, 4>::random(shape, dist, &device);

        let weight = self.kernel_posterior_loc.val() + (sigma * epsilon);

        let options = ConvOptions::new(self.stride, self.padding, [1,1], 1);

        burn::tensor::module::conv2d(
            input,
            weight,
            Some(self.bias_posterior_loc.val()),
            options,
        )
    }
}

impl DenseFlipoutConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DenseFlipout<B> {
        let shape = [self.input_size, self.output_size];
        DenseFlipout{
            kernel_posterior_loc: Param::from_tensor(Tensor::zeros(shape.clone(), device)),
            kernel_posterior_untransformed_scale: Param::from_tensor(Tensor::zeros(shape, device)),
            bias_posterior_loc: Param::from_tensor(Tensor::zeros([self.output_size], device)),
        }
    }
}

impl<B: Backend> DenseFlipout<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = self.kernel_posterior_loc.device();
        let shape = self.kernel_posterior_loc.shape();

        let sigma = softplus(self.kernel_posterior_untransformed_scale.val(), 1.0);
        let epsilon = Tensor::<B, 2>::random(shape, Distribution::Normal(0.0, 1.0), &device);
        let weight = self.kernel_posterior_loc.val() + (sigma * epsilon);

        input.matmul(weight) + self.bias_posterior_loc.val().unsqueeze()
    }
}