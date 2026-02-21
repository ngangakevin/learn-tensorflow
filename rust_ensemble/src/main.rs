mod model;
mod layers;
mod utils;

use safetensors::SafeTensors;
use memmap2::MmapOptions;
use std::fs::File;
use std::io::{stdout, BufWriter};
use burn_ndarray::NdArray;
use burn::tensor::{Tensor};
use model::LeNet5BNNConfig;
use ferris_says::say;

fn main() {
    say("BNN: Convolutional Network brought to you by Sahihi", 32, BufWriter::new(
        stdout().lock())).unwrap();
    type MyBackend = NdArray<f32, i32>;
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let filename = "../model.safetensors";
    let file = File::open(filename).unwrap_or_else(|_| {
        panic!("Could not find model.safetensors at {}. Check your current directory!", filename)
    });
    let buffer = unsafe { MmapOptions::new().map(&file).expect("Could not open mmap file") };
    let tensors = SafeTensors::deserialize(&buffer).expect("Could not deserialize tensors");

    println!("--- Listing SafeTensor Keys");
    let names = tensors.names();
    for name in names {
        let view = tensors.tensor(name).unwrap();
        println!("Key: \"{}\" | Shape: {:?} | Dtype: {:?}", name, view.shape(), view.dtype());
    }
    println!("------------------------- ");

    let config = LeNet5BNNConfig::new(10, 1);
    let mut model = config.init::<MyBackend>(&device);

    println!("Loading weights (CPU Mode)...");

    model.load_weights(&tensors, &device);

    println!("Model loaded successfully on CPU!");

    let input = Tensor::<MyBackend, 4>::zeros([1,1,28,28], &device);

    let num_samples = 10;
    let mut aggregated_predictions = Tensor::<MyBackend, 2>::zeros(
        [1, num_samples], &device);
    let mut results: Vec<Tensor<MyBackend, 2>> = Vec::new();

    println!("Running {} Monte Carlo Samples...", num_samples);

    for _ in 0..num_samples {
        let output = model.forward(input.clone());
        aggregated_predictions = aggregated_predictions.clone() + output;

        let mean_prediction = aggregated_predictions.clone() / (num_samples) as f32;
        println!("\nFinal Mean Prediction Probabilities:\n{}", mean_prediction);
    }

    for _ in 0..num_samples {
        let output = model.forward(input.clone());
        results.push(output);
    }

    let stacked_results:Tensor<MyBackend, 3> = Tensor::<MyBackend, 2>::stack(results, 0);

    let mean_prediction = stacked_results.clone().mean_dim(0);
    let mean_of_squares = stacked_results.powf_scalar(2.0).mean_dim(0);
    let squared_of_mean = mean_prediction.clone().powf_scalar(2.0);

    let variance = mean_of_squares - squared_of_mean;
    let std_dev = (variance).sqrt();

    println!("\n --- Results ---");
    println!("Mean Probabilities:\n{}", mean_prediction);
    println!("Uncertainty (Std Dev):\n{}", std_dev);

}
