# Experiments on SignGD-Based Neuronal Dynamics

This directory contains all the necessary scripts and configurations to evaluate the performance of the **sign gradient descent (signGD)-based neuronal dynamics** and **subgradient-based neuronal dynamics** in ANN-to-SNN conversion.

---

## Running Experiments

### 1. Training an ANN Model

To train an ANN model that will later be converted to an SNN, use the following command:

```bash
python -m scripts.train --config [config_path] --gpu [gpu_number]
```

- **`config_path`**: Path to the experiment configuration file (e.g., `config/signgd.py` or `config/subgradient.py`).
- **`gpu_number`**: The GPU device to be used for training (e.g., `0` for the first GPU).

---

### 2. Evaluating the Trained Model (ANN-to-SNN Conversion)

Once the ANN model is trained, you can convert it into an SNN and evaluate its performance on an image classification dataset using the following command:

```bash
python -m scripts.evaluate --config [config_path] --gpu [gpu_number]
```

- **`config_path`**: Path to the configuration file used during training (e.g., `config/signgd.py`).
- **`gpu_number`**: The GPU device to be used for evaluation (e.g., `0` for the first GPU).

#### Example: Evaluating SignGD-Based Neuronal Dynamics

To convert the trained ANN model into an SNN using the **signGD-based neuronal dynamics** and evaluate it on an image classification dataset, use:

```bash
# ANN-to-SNN conversion and evaluation using signGD-based dynamics
python -m scripts.evaluate --config config/signgd.py --gpu 0
```

#### Example: Evaluating Subgradient-Based Neuronal Dynamics

Similarly, to evaluate a model trained using the **subgradient-based neuronal dynamics**, run:

```bash
# ANN-to-SNN conversion and evaluation using subgradient-based dynamics
python -m scripts.evaluate --config config/subgradient.py --gpu 0
```


## Example Workflow

Here is an example workflow that demonstrates how to train a model using signGD-based neuronal dynamics and evaluate it through ANN-to-SNN conversion:

1. **Train the ANN model**:

    ```bash
    python -m scripts.train --config config/signgd.py --gpu 0
    ```

2. **Convert the trained ANN to SNN and evaluate**:

    ```bash
    python -m scripts.evaluate --config config/signgd.py --gpu 0
    ```

## Experiment Configurations

- **`config/signgd.py`**: Contains configurations specific to the experiments on **signGD-based neuronal dynamics**.
- **`config/subgradient.py`**: Contains configurations for the **subgradient-based neuronal dynamics**.

These configurations include important parameters such as learning rate, batch size, optimizer settings, and network architecture details.

## Directory Overview

- **config/**: Contains experiment configurations for both **signGD-based neuronal dynamics** and **subgradient-based neuronal dynamics**.
- **logs/**: Stores log files generated during training and evaluation.
- **resources/**: Includes checkpoints, pretrained models, and other experiment-related resources.
- **scripts/**: Python scripts for training and evaluating models.
  - **train.py**: Script for training artificial neural networks (ANNs) using the configurations.
  - **evaluate.py**: Script for converting the trained ANNs into spiking neural networks (SNNs) and evaluating their performance.

## Notes

1. **GPU Usage**: Make sure to specify the appropriate `--gpu` argument based on the GPU device available for training and evaluation.
2. **Customization**: You can modify the experiment configurations (e.g., learning rate, model architecture) inside the **config/** directory to suit your research needs.
