# Speech Quantization Experiments for Language Identification

## Overview

This repository contains the code developed for a project on **closed-set spoken Language Identification (LID)**. The project investigates how different **audio quantization schemes** affect the performance of both classical machine learning models and deep learning models.

Experiments were performed using the **Google FLEURS** multilingual speech dataset, with both binary (2-language) and multiclass (10-language and 102-language) language identification tasks.

The repository includes implementations using:

- Gaussian Mixture Models (GMM)
- Support Vector Machines (SVM)
- Logistic Regression
- Naive Bayes
- ECAPA-TDNN (SpeechBrain)

---

# Project Objectives

The primary objective of this work is to study the effect of different quantization levels on speech-based language identification.

The experiments compare:

- 16-bit (original audio)
- 8-bit quantization
- 4-bit quantization
- 2-bit quantization
- 1-bit quantization

The repository also contains experiments evaluating mismatched train-test quantization conditions (for example, training on 16-bit audio and testing on 1-bit audio).

---

# Repository Structure

```
├── GMM-UBM/
├── Logistic Regression/
├── Naive Bayes/
├── SVM/
├── TDNN/
│   └── Fleurs/
│       ├── src/
│       ├── configs/
│       ├── data_preparation/
│       ├── 10Classes/
│       ├── Multi_ClassWith1bit/
│       └── Iteration_2_1bit_Quantization/
└── README.md
```

---

# Classical Machine Learning Models

Each classical model directory contains the implementation used for the corresponding experiments.

- Feature extraction
- Audio quantization
- Model training
- Evaluation

Models include:

- GMM
- Logistic Regression
- Naive Bayes
- SVM

---

# TDNN (SpeechBrain)

The TDNN implementation is based on the SpeechBrain ECAPA-TDNN recipe and has been adapted for multilingual language identification on the Google FLEURS dataset.

The folder has been organized so that reusable code is separated from experiment-specific configurations.

```
TDNN/
└── Fleurs/
    ├── src/
    ├── configs/
    ├── data_preparation/
    ├── 10Classes/
    ├── Multi_ClassWith1bit/
    └── Iteration_2_1bit_Quantization/
```

---

# Folder Description

## src/

Contains the common source files used by all experiments.

### train.py

Main SpeechBrain training script.

Responsible for:

- loading datasets
- initializing ECAPA-TDNN
- training
- validation
- testing
- checkpointing

This script is shared across all experiments.

---

### speech_quantization.py

Implements the audio quantization routines used throughout the experiments.

Supports multiple quantization schemes including:

- 16-bit
- 8-bit
- 4-bit
- 2-bit
- 1-bit

---

## configs/

Contains YAML configuration files for different experiments.

Example:

```
configs/
    fleurs_16bit.yaml
    fleurs_8bit.yaml
    fleurs_4bit.yaml
    fleurs_2bit.yaml
    fleurs_1bit.yaml
```

Each YAML file specifies:

- dataset paths
- output directories
- model hyperparameters
- optimizer settings
- feature extraction parameters
- number of languages
- quantization configuration

Only the configuration changes between experiments; the training script remains the same.

---

## data_preparation/

Contains scripts used for preparing the Google FLEURS dataset.

Example:

- creation of WebDataset shards
- metadata generation
- preprocessing

---

## 10Classes/

Contains experiments performed on a 10-language subset of Google FLEURS.

Subdirectories correspond to different quantization schemes.

Examples include:

- 16Bit
- 8Bit
- 4Bit
- 2Bit
- 1Bit

Cross-quantization experiments such as:

- Train on 16-bit → Test on 1-bit
- Train on 1-bit → Test on 16-bit

are also included.

---

## Multi_ClassWith1bit/

Contains experiments performed on the full multilingual dataset using 1-bit quantized audio.

---

## Iteration_2_1bit_Quantization/

Contains an earlier iteration of the 1-bit quantization experiments retained for reference.

---

# Running Experiments

## Step 1

Prepare the Google FLEURS dataset.

Run the dataset preparation script inside

```
data_preparation/
```

to create the required metadata and WebDataset shards.

---

## Step 2

Choose the desired experiment configuration.

Example:

```
configs/fleurs_4bit.yaml
```

or

```
configs/fleurs_16bit.yaml
```

---

## Step 3

Run the training script.

Example:

```bash
python src/train.py configs/fleurs_4bit.yaml
```

The training script reads the selected YAML configuration and automatically performs training using the specified quantization scheme.

---

# SpeechBrain

The TDNN experiments are built using:

- SpeechBrain
- PyTorch

Model:

- ECAPA-TDNN

Features:

- Log Mel Filterbanks (FBank)

---

# Dataset

Google FLEURS

The dataset contains multilingual speech recordings used for language identification experiments.

---

# Notes

Only the final versions of the common source files have been retained.

Experiment-specific settings are stored in their corresponding YAML configuration files inside the `configs/` directory.

Training outputs, checkpoints, logs, cache files, and generated artifacts are excluded from this repository.
