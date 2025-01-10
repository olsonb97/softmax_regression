
# Softmax Regression Foundations

![Softmax Model Analytics](res/demo.png)
A minimal implementation of softmax regression using only NumPy and Matplotlib. This project is designed for experimenting and understanding how softmax works under the hood.
It utilizes multinomial regression to model the feature space, allowing for any number of classes and features.

## Overview

- **model.py**: Contains the `SoftmaxModel` class, which handles training, inference, and saving/loading model parameters.

- **plot.py**: Contains the `SoftmaxPlot` class for plotting training analytics. Can be enabled or disabled as a popup.

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/olsonb97/softmax_regression.git
   ```
2. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```

## Usage

1. Prepare dataset and labels. Normalize the data, as the model does not handle this internally.
2. Create an instance of `SoftmaxModel` and call its `train` method:
   ```python
   from model import SoftmaxModel

   model = SoftmaxModel()
   model.train(
       dataset, 
       labels, 
       epochs=50, 
       batches=5, 
       learning_rate=0.01, 
       decay_rate=0.001, 
       shuffle=True, 
       plot=True
   )
   ```
3. Evaluate performance:
   ```python
   accuracy = model.test(dataset, labels)
   print("Accuracy:", accuracy)
   ```
4. Save model parameters:
   ```python
   model.save("params.npz")
   ```
5. Load model parameters and avoid re-training:
   ```python
   model = SoftmaxModel()
   model.load("params.npz")
   ```

## Notes

- The `SoftmaxModel` class inherits from `SoftmaxPlot` to allow plotting the training metrics.
- This project is intended for educational/experimental purposes and runs quite slow compared to better suited alternatives, such as Sci-kit.
