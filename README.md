# Neural Arithmetic: Sequence-to-Sequence LSTM for Integer Addition

## Overview

This project demonstrates how a sequence-to-sequence (Seq2Seq) LSTM model can be trained to learn and perform basic integer addition by treating it as a character-level translation task. Inspired by how models learn language translation, this implementation showcases how deep learning can generalize mathematical operations when trained on formatted input-output examples.

## Key Features

* Character-level sequence processing
* LSTM-based encoder-decoder architecture
* Reversed input sequences to improve learning
* Expanded data complexity (3-digit addition)
* Vectorized input/output with one-hot encoding
* Training visualizations using loss curves

## Technologies Used

* Python 3.x
* TensorFlow / Keras
* NumPy
* Matplotlib

## Dataset

Synthetic data is generated dynamically using random integer pairs. Each pair is represented as a string (e.g., "345+678") and the target output is the result of the addition, also in string format (e.g., "1023").

## Model Architecture

* LSTM Encoder (128 units)
* RepeatVector for decoder input
* LSTM Decoder (128 units, return sequences)
* TimeDistributed Dense layer with softmax activation

## Getting Started

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/lstm-addition-model.git
   cd lstm-addition-model
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:

   ```bash
   python train_addition_model.py
   ```

## Results

The model is able to predict the sum of two integers with high accuracy after training on a sufficiently large dataset of synthetic addition problems. Evaluation includes a side-by-side comparison of true vs predicted output.

## Example Output

```
Q: 435+219 | T:  654 | P:  654
Q:  12+897 | T:  909 | P:  909
Q:  56+432 | T:  488 | P:  488
```
