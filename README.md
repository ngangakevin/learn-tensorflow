
## Technology Stack

- **Python (66%)**: Jupyter Notebooks, TensorFlow, Keras
  - Deep learning model development
  - Bayesian Neural Networks
  - Data analysis and visualization
  
- **Rust (34%)**: High-performance inference
  - Model ensemble implementation
  - Efficient inference pipelines

## Key Dependencies

### Core ML Libraries
- **TensorFlow 2.20.0**: Deep learning framework
- **Keras 3.13.2**: High-level neural networks API
- **TensorFlow Probability 0.25.0**: Probabilistic modeling

### Supporting Libraries
- **NumPy 2.4.1**: Numerical computing
- **Pandas 3.0.0**: Data manipulation
- **Scikit-learn 1.8.0**: Machine learning utilities
- **Matplotlib 3.10.8**: Data visualization
- **Jupyter Lab 4.5.3**: Interactive notebooks

## Getting Started

### Prerequisites
- Python 3.8+
- Rust (for rust_ensemble features)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ngangakevin/learn-tensorflow.git
cd learn-tensorflow
```
2. Install Python dependencies:
```bash
pip install -r requirements.txt
```
3. (Optional) Build Rust components:
```bash
cd rust_ensemble
cargo build --release
```

 ### Running the Project
 1. Launch Jupyter Lab:
 ```bash
jupyter lab
```
2. Open bnn.ipynb to explore the Bayesian Neural Network implementation.
3. Load trained models:
```bash
import keras
model = keras.models.load_model('final_model.keras')
```
### Project Components

## Bayesian Neural Networks (bnn.ipynb)

- Implementation of probabilistic neural networks
- Uncertainty quantification in predictions
- Training and evaluation notebooks
- Model visualization and analysis
- Trained Models

final_model.keras: Keras format model for easy loading in TensorFlow
model.safetensors: SafeTensors format for interoperability
Rust Ensemble Module

High-performance inference and model ensembling using Rust for better performance and memory efficiency.

## Usage Examples

Loading and using the trained model:
```python
import tensorflow as tf
import keras

# Load the model
model = keras.models.load_model('final_model.keras')

# Make predictions
predictions = model.predict(your_data)
```
### Development Workflow

This project uses Jupyter Notebooks for interactive development and experimentation. The bnn.ipynb notebook contains the main implementation details and can be used to:

Train models
Evaluate performance
Visualize results
Experiment with different architectures
Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

Please check the repository for license information.

## Author

Kevin Nganga (@ngangakevin)

Last Updated: February 2026
