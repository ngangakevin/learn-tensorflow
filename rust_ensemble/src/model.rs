use burn::module::{Module, Param};
use burn::{Tensor};
use burn::config::Config;
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::tensor::backend::Backend;
use crate::layers::{Conv2dFlipout, Conv2dFlipoutConfig, DenseFlipout, DenseFlipoutConfig};
use crate::utils;

#[derive(Config, Debug)]
pub struct LeNet5BNNConfig {
    pub num_classes: usize,
    pub input_channels: usize,
}

#[derive(Module, Debug)]
pub struct LeNet5BNN<B: Backend> {
    pub conv1: Conv2dFlipout<B>,
    pub pool1: MaxPool2d,
    pub conv2: Conv2dFlipout<B>,
    pub pool2: MaxPool2d,
    pub conv3: Conv2dFlipout<B>,
    pub fc1: DenseFlipout<B>,
    pub fc2: DenseFlipout<B>,
}


impl LeNet5BNNConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LeNet5BNN<B>{
        LeNet5BNN {
            conv1: Conv2dFlipoutConfig::new([self.input_channels, 6], [5, 5]).init(device),
            pool1: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            conv2: Conv2dFlipoutConfig::new([6, 16], [5, 5]).init(device),
            pool2: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            conv3: Conv2dFlipoutConfig::new([16, 120], [5, 5]).init(device),
            fc1: DenseFlipoutConfig::new(5880, 84).init(device),
            fc2: DenseFlipoutConfig::new(84, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> LeNet5BNN<B>{
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2>{
        let x = self.conv1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.pool1.forward(x);

        let x = self.conv2.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.pool2.forward(x);

        let x = self.conv3.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = x.flatten::<2>(1,3);

        let x = self.fc1.forward(x);
        let x = burn::tensor::activation::relu(x);

        let x = self.fc2.forward(x);
        burn::tensor::activation::softmax(x, 1)
    }
}

impl<B: Backend> LeNet5BNN<B>{
    fn map_conv(
        prefix: &str,
        layer: &mut Conv2dFlipout<B>,
        tensors: &safetensors::SafeTensors,
        device: &B::Device,
    ){
        layer.kernel_posterior_loc = Param::from_tensor(utils::load_tensor(
            &format!("{}.kernel_mu", prefix), tensors, device));
        layer.kernel_posterior_untransformed_scale = Param::from_tensor(utils::load_tensor(
            &format!("{}.kernel_rho", prefix), tensors, device));
        layer.bias_posterior_loc = Param::from_tensor(utils::load_flat_tensor::<B, 1>(
            &format!("{}.bias", prefix), tensors, device));
    }

    fn map_dense(
        prefix: &str,
        layer: &mut DenseFlipout<B>,
        tensors: &safetensors::SafeTensors,
        device: &B::Device,
    ) {
        layer.kernel_posterior_loc = Param::from_tensor(utils::load_flat_tensor::<B, 2>(
            &format!("{}.kernel_mu", prefix), tensors, device));
        layer.kernel_posterior_untransformed_scale = Param::from_tensor(
            utils::load_flat_tensor::<B, 2>(&format!("{}.kernel_rho", prefix), tensors, device));
        layer.bias_posterior_loc = Param::from_tensor(utils::load_flat_tensor::<B, 1>(
            &format!("{}.bias", prefix), tensors, device))
    }
    pub fn load_weights(&mut self, tensors: &safetensors::SafeTensors, device: &B::Device){
        let all_names: Vec<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();

        let get_sorted_prefixes = |pattern: &str| -> Vec<String> {
            let mut prefixes: Vec<String> = all_names.iter()
                .filter(|name| name.contains(pattern))
                .map(|name| name.split('.').next().unwrap().to_string())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            prefixes.sort_by_key(|s| {
                s.chars().filter(|c| c.is_digit(10))
                    .collect::<String>()
                    .parse::<i32>().unwrap_or(0)
            });
            prefixes
        };

        let conv_prefixes = get_sorted_prefixes("conv2d_flipout");
        let dense_prefixes = get_sorted_prefixes("dense_flipout");

        if conv_prefixes.len() >= 3 {
            Self::map_conv(&conv_prefixes[0], &mut self.conv1, tensors, device);
            Self::map_conv(&conv_prefixes[1], &mut self.conv2, tensors, device);
            Self::map_conv(&conv_prefixes[2], &mut self.conv3, tensors, device);
        }

        if dense_prefixes.len() >= 2 {
            Self::map_dense(&dense_prefixes[0], &mut self.fc1, tensors, device);
            Self::map_dense(&dense_prefixes[1], &mut self.fc2, tensors, device);
        }
    }
}