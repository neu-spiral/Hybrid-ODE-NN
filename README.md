# README

## A Hybrid ODE-NN Framework for Modeling Incomplete Physiological Systems

### Overview
This repository provides the implementation of a novel method to learn approximations of missing Ordinary Differential Equations (ODEs) and states in physiological models where the system's states and dynamics are partially unknown. The approach combines the use of ODEs and neural networks (NNs) to approximate unknown system dynamics while leveraging recursive Bayesian estimation to jointly estimate NN parameters and unknown initial conditions during training. The proposed framework aims to model physiological systems where some of the differential equations and state information are incomplete or not directly measurable.

### Features
- Hybrid ODE and Neural Network (NN) modeling.
- Recursive Bayesian estimation for joint estimation of latent states, NN parameters, and initial conditions.
- Cubature Kalman Filtering (CKF) to tackle non-linearity in the model and to approximate intractable integrals efficiently.
- Application on physiological systems like the Hodgkin-Huxley neuron model and a compartmental model for retinal circulation.
- Robustness to noisy measurements and system input perturbations.

### Code Description
- **cubature_filter.py**: Implements the cubature Kalman filter for state and parameter estimation, using a cubature rule for integral approximations.
- **neural_ode_estimation.py**: Contains the core implementation of the hybrid ODE-NN model. The model uses a combination of known ODE equations and neural networks for missing states.
- **helper_functions.py**: Utility functions to assist in cubature point generation, matrix operations, and related computations.
- **hh_kalman.py**: Script that integrates the Hodgkin-Huxley model with Kalman filtering for hybrid state estimation.
- **tests/**: A directory with sample scripts for testing the hybrid modeling approach on simulated physiological systems.

### Prerequisites
To run the code, ensure that the following libraries are installed:
- Python 3.8+
- TensorFlow 2.8+
- NumPy
- Matplotlib
- SciPy

Install all dependencies using the provided requirements file:
```sh
pip install -r requirements.txt
```

### Getting Started
Follow these steps to get started:
1. Clone the repository:
    ```sh
    git clone https://github.com/username/hybrid_ode_nn.git
    cd hybrid_ode_nn
    ```
2. Install the required packages using the command:
    ```sh
    pip install -r requirements.txt
    ```
3. Run a test on one of the case studies:
    ```sh
    python tests/neuron_model_test.py
    ```

### Usage
- The **CubatureFilter** class can be used to estimate parameters and missing states of a hybrid model.
- You can adapt the **neural_ode_estimation.py** to your own physiological model by specifying the known ODEs and initializing the neural network to approximate missing dynamics.
- Modify parameters like process noise covariance (Q), measurement noise covariance (R), and initial state covariance (P0) to match your system characteristics.

### Example Code Usage for Hodgkin-Huxley Model
The following is an example of how to use the hybrid ODE-NN framework for the Hodgkin-Huxley model:

```python
import numpy as np
import tensorflow as tf
from hh_kalman import hh_model, CubatureFilter

# Define initial conditions and parameters
initial_conditions = [-65.0, 0.3, 0.05, 0.6]  # V, n, m, h
step_size = 0.01
is_learnable = [True, True, True, True]

# Create the Hodgkin-Huxley model
hh = hh_model(is_learnable=is_learnable, ie=10, descr='gsa preset', step_size=step_size)

# Define the state transition function (f) and measurement function (h)
def f(z, input):
    return hh.forward(input, states=z[-len(initial_conditions):])

def h(z):
    observation_matrix = np.zeros((np.sum(is_learnable), 4))
    for index, idx in enumerate(np.where(is_learnable)[0]):
        observation_matrix[index, idx] = 1
    return np.dot(observation_matrix, z[-4:])

# Initialize the cubature Kalman filter
x0 = np.concatenate((np.random.rand(hh.num_of_learnables), initial_conditions))
P0 = np.eye(len(x0)) * 0.1
Q0 = np.eye(len(x0)) * 0.01
R0 = np.eye(len(initial_conditions)) * 0.1
nncbf = CubatureFilter(f=f, h=h, x0=x0, P0=P0, Q0=Q0, R0=R0, s_dim=len(initial_conditions))

# Run a prediction and update step
u = np.array([0.5, 0.2])  # Example input
predicted_state, predicted_covariance = nncbf.predict(sigmoid=None, u=u)
print("Predicted State:", predicted_state)
```

### Directory Structure
- **cubature_filter.py**: Main file containing cubature Kalman filtering logic for state and parameter estimation.
- **neural_ode_estimation.py**: Defines the hybrid ODE-NN model framework and training logic.
- **helper_functions.py**: Utility functions to generate cubature points and perform matrix operations.
- **hh_kalman.py**: Script for integrating Hodgkin-Huxley neuron modeling with Kalman filtering.
- **tests/**: Contains test scripts that simulate different physiological systems and test the hybrid modeling approach.

### Applications
- **Hodgkin-Huxley Neuron Model**: Tests the framework's ability to accurately recover missing neuron dynamics and membrane potential under noisy conditions.
- **Retinal Circulation Model**: Demonstrates the effectiveness of the hybrid model in estimating internal pressures of multiple compartments in the retina.

### Results
The proposed approach was evaluated using Monte Carlo experiments on simulated physiological models. The approach showed the ability to:
- Accurately track missing states when a subset of the system of ODEs is unknown.
- Demonstrate robustness to input perturbations and noisy measurements.
- Exhibit flexibility in learning unknown dynamics while keeping the known parts of the model intact.

### Limitations
The proposed approach assumes knowledge of state dependencies, which may not be available in some cases. The accuracy of the neural network components also depends on the quality of training data and the chosen hyperparameters.

### Contact
For any inquiries, please contact [author's email].

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
