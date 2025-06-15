# Micro-Expression Detection for Lie Detection

This project aims to detect micro-expressions related to lying in facial images using deep learning techniques. The system analyzes subtle facial expressions to determine whether a person is telling the truth or lying.

## Dataset

The dataset contains facial images of different participants answering questions in two conditions:
- **Truth**: When participants are telling the truth
- **Lie**: When participants are deliberately lying

The data is organized by participant and question type, with separate directories for training and testing.

## Project Structure

```
.
├── data/                  # Raw data directory
├── src/                   # Source code
│   ├── models/            # Neural network model definitions
│   ├── data/              # Data loading and processing utilities
│   ├── utils/             # Utility functions
│   ├── config.py          # Configuration parameters
│   ├── train.py           # Training script
│   └── test.py            # Testing and evaluation script
├── data_splits/           # Processed data splits
├── models/                # Saved trained models
├── checkpoints/           # Training checkpoints
├── logs/                  # Training logs and visualizations
├── prepare_dataset.py     # Script for preparing the dataset
└── run.py                 # Main entry script
```

## Installation



1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Setting up the project

```bash
python run.py setup
```

This will create all necessary directories for the project.

### Preparing the dataset

```bash
python run.py prepare
```

This will process the raw data, extract labels and create train/test splits.

### Preprocessing faces

```bash
python run.py preprocess
```

This will detect and align faces in all images for training and testing.

### Visualizing dataset

```bash
python run.py visualize
```

This will generate statistics and visualizations of the dataset, saving them to the logs directory.

### Training a model

```bash
python run.py train --model cnn --epochs 10 --batch_size 32
```

Available model types:
- `cnn`: Custom CNN model
- `mobilenet`: MobileNetV2 transfer learning model
- `efficientnet`: EfficientNetB0 transfer learning model

### Testing a trained model

```bash
python run.py test --model models/custom_cnn_final.pt --model_type cnn
```

Or simply use the model name:

```bash
python run.py test --model custom_cnn_final --model_type cnn
```


## Face Detection and Preprocessing

The system uses the MTCNN (Multi-Task Cascaded Convolutional Neural Network) for face detection and alignment. Following detection, facial segmentation is performed to isolate key facial regions, which can aid in more focused micro-expression analysis. During the preprocessing step, you also have the option to visualize these segmented facial areas to inspect the results. You can also turn off the segmentation if you decide to do so from the config python file.

## Evaluation Metrics

Models are evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC (Area Under ROC Curve)
- Confusion Matrix

## Results

Results of model training and evaluation are stored in the `logs/` directory, including:
- Training history plots
- Confusion matrices
- ROC curves
- Test metrics 