
# Image Classification Challenge Project

## Overview

This repository contains code for IEEE-CS Enrollment Task's Image Classification Challenge where you classify images (70,000 grayscale images of 28×28 pixels), representing "fashion items" into 10 categories. The challenge is divided into these 4 task levels:

1. **Level 0: Data Loading and Initial Inspection**  
   Load the dataset, print its shape, and display a few sample images with their labels.

2. **Level 1: Exploratory Data Analysis (EDA)**  
   Perform data visualization: show one sample image per category, and display pixel histograms.

3. **Level 2: Basic Classification Model**  
   Train a basic Logistic Regression model, evaluate its accuracy.

4. **Level 3: Neural Network Implementation**  
   Train a simple neural network to boost accuracy.

## Downloading the Dataset

The dataset can be downloaded from [here](https://drive.google.com/file/d/1byxncPUl2aeKFZ0voFAQ7WbyjBSvLhNA/view?usp=sharing).  
- **Note:** Unzip it and place it in the root of the repository.
- If you prefer to use your own dataset, ensure it follows this format:
  - The first column is the label (an integer value).
  - The next 784 columns contain pixel values (since the images are 28px*28px).
 
## Downloading the required libraries

- Ensure that you have pip installed.
- Open terminal the repository where all files are saved.
- Run the following command:
  ```bash
  pip install -r requirements.txt
  ```


## Code Files

The repository contains 4 Python files, one per task level. You can run each file separately using a terminal (Git Bash, Command Prompt, etc.).

### Level 0: Data Loading and Initial Inspection
- **File:** `level_0_data_loading.py`
- **Purpose:**  
  Loads the dataset using pandas, prints its shape, and displays a few sample images with their labels.
- **Usage:**  
  ```bash
  python level_0_data_loading.py


### Level 1: Exploratory Data Analysis (EDA)
- **File:** `level_1_exploratory_data_analysis.py`
- **Purpose:**  
  Loads the dataset, prints summary statistics, plots label distributions, shows one sample image per class, and displays a histogram of pixel intensities.
- **Usage:**  
  ```bash
  python level_1_exploratory_data_analysis.py
  ```

### Level 2: Basic Classification Model (Logistic Regression)
- **File:** `level_2_logistic_regression.py`
- **Purpose:**  
  Preprocesses and normalizes the data, splits it into training and testing sets, trains a Logistic Regression model, evaluates its accuracy, displays a confusion matrix, and shows the learned coefficients as images.
- **Usage:**  
  ```bash
  python level_2_logistic_regression.py
  ```

### Level 3: Neural Network Implementation
- **File:** `level_3_neural_network.py`
- **Purpose:**  
  Implements a neural network using PyTorch. The code reshapes flat image vectors to 28×28 images, trains the network (with optional GPU support), and displays sample predictions.
- **Usage:**  
  ```bash
  python level_3_neural_network.py
  ```

## Running on GPU vs. CPU

- The neural network code checks for a CUDA-enabled GPU:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
- If you have a dedicated GPU (e.g., NVIDIA), install the GPU-enabled version of PyTorch. Verify with:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
  If `True`, it will run on your GPU; otherwise, it will run on CPU.

## If you want to Speed Up Training on your machine

- **Increase Batch Size:**  
  In the DataLoader (e.g., in `level_3_neural_network.py`), you can increase the `batch_size` if your GPU has enough memory.
- **Mixed Precision:**  
  Use PyTorch’s mixed precision (`torch.amp.autocast`) for faster training on supported GPUs.
- **DataLoader Workers:**  
  Increase the `num_workers` parameter to improve data loading speed if your CPU isn’t the bottleneck.
- **Hyperparameter Tuning:**  
  Experiment with learning rates, dropout rates, and the number of epochs.

## Final Notes

- This repository is designed for educational purposes/ as a task. Feel free to modify and extend the code.
- Please reach out to @ludicrouslytrue on discord for any help
