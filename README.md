# Product detection using TensorFlow

A project detect products in general and support payments in particular serving supermarket systems.

# Overview

<img src="https://i.imgur.com/iViRxxP.png"> 

<img src="https://i.imgur.com/Qx6QePk_d.jpg?maxwidth=520&shape=thumb&fidelity=high">

This repository contains a comprehensive solution for item recognition in supermarket systems using TensorFlow. The goal of this project is to streamline the checkout process by automatically identifying items and supporting payment processing, thus reducing the need for human cashiers and enhancing operational efficiency.

## Recognition Logic:

Tracks whether confident score above
99.99% recognition has been maintained for
at least 2 seconds.

If the recognition is stable and confident, it
adds the recognized class to a "cart" (a
dictionary that tracks the count of each
recognized item).

Ending Video Capture: When the user
presses 'q'
, it stops the video capture and
closes the display window

## Model using

<img src="https://i.imgur.com/tqIAtUZ.png">

EfficientNet, introduced by Mingxing Tan and Quoc V. Le in 2019, is a family of models that aim to balance
network depth, width, and resolution for efficient scaling. The key innovation behind EfficientNet is the
compound scaling method, which uniformly scales all dimensions of depth, width, and resolution using a set
of fixed scaling coefficients.

## Key Features

1.Item Recognition:

- Utilizes TensorFlow for robust and accurate item recognition.
- Trained on a custom dataset of supermarket items to ensure high accuracy.
- Capable of recognizing items from various angles and lighting conditions.

  2.Real-time Processing:

- Optimized for real-time item recognition to minimize checkout time.
- Uses TensorFlow Serving for efficient model deployment and inference.

  3.Seamless Integration:

- Easily integrates with existing supermarket POS systems.
- Part of the automation system.

4. Scalability:

- Designed to handle high volumes of transactions.
- Can be deployed on-premise or in the cloud.

# Getting Started

## Prerequisites

- TensorFlow 2.x
- Python 3.7+
- Necessary libraries listed in requirements.txt

## Installation

- Step-by-step guide to set up the development environment.

```bash
conda create ur_env python==3.9
conda activate ur_env
pip install -n "requirement.txt"
```

- Instructions for installing dependencies and configuring the system.

## Usage

- Example scripts for running item recognition.

```
python script_name.py --model_path "model.weight.h5" --database_path "ur_database.csv"
```

# Contributions

Contributions to this project are welcome. Please submit a pull request with your changes.

# License

This project is licensed under the MIT License. See the LICENSE file for details.
