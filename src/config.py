"""Configuration parameters for the micro-expression detection project"""
import os

# Data Processing
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
FACE_MARGIN = 0.2

# Model Training
LEARNING_RATE = 0.001
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.2
MIN_LR = 1e-6

# Model Architecture
CNN_FILTERS = [32, 64, 128, 256]
DENSE_UNITS = 512
DROPOUT_RATES = [0.5, 0.3]

# Paths
DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'Train', 'Train')
TEST_DIR = os.path.join(DATA_DIR, 'Test', 'Test')

# Directories for saving and loading files
DATA_SPLITS_DIR = 'data_splits'
MODELS_DIR = 'models'
CHECKPOINTS_DIR = 'checkpoints'
FACE_EMBEDDINGS_DIR = 'face_embeddings'
LOGS_DIR = 'logs'
TRAIN_CACHE_DIR = 'train_face_cache'
TEST_CACHE_DIR = 'test_face_cache'

# Face mask directories
FACE_MASKS_DIR = 'face_masks'
TRAIN_MASK_DIR = os.path.join(FACE_MASKS_DIR, 'train')
TEST_MASK_DIR = os.path.join(FACE_MASKS_DIR, 'test')

# Visualization Settings
VISUALIZATION_DIR = 'visualizations'
SAVE_SEGMENTATION_SAMPLES = True  # Enable saving segmented image samples
NUM_VISUALIZATION_SAMPLES = 20  # Number of samples to save

# Segmentation Settings
USE_SEGMENTATION = True  # Enable real-time face region segmentation 