import os
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from pathlib import Path
from src.config import USE_SEGMENTATION

def get_train_test_paths_and_labels():
    """Get train and test image paths with their corresponding labels from predefined splits."""
    base_path = 'data'
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    
    # Process Train directory
    print("Processing training data...")
    train_path = os.path.join(base_path, 'Train', 'Train')
    for label in ['Lie', 'Truth']:
        label_path = os.path.join(train_path, label)
        for participant in tqdm(os.listdir(label_path), desc=f"Processing {label} samples"):
            participant_path = os.path.join(label_path, participant)
            for question_dir in os.listdir(participant_path):
                question_path = os.path.join(participant_path, question_dir)
                for img_file in glob(os.path.join(question_path, '*.png')):
                    train_paths.append(img_file)
                    train_labels.append(1 if label == 'Lie' else 0)
    
    # Process Test directory
    print("\nProcessing test data...")
    test_path = os.path.join(base_path, 'Test', 'Test')
    for label in ['Lie', 'Truth']:
        label_path = os.path.join(test_path, label)
        for participant in tqdm(os.listdir(label_path), desc=f"Processing {label} samples"):
            participant_path = os.path.join(label_path, participant)
            for question_dir in os.listdir(participant_path):
                question_path = os.path.join(participant_path, question_dir)
                for img_file in glob(os.path.join(question_path, '*.png')):
                    test_paths.append(img_file)
                    test_labels.append(1 if label == 'Lie' else 0)
    
    # Convert labels to PyTorch tensors
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)
    
    return {
        'train': (np.array(train_paths), train_labels),
        'test': (np.array(test_paths), test_labels)
    }

def main():
    # Create data_splits directory if it doesn't exist
    os.makedirs('data_splits', exist_ok=True)
    
    # Get train and test paths and labels
    print("Collecting image paths and labels from predefined splits...")
    splits = get_train_test_paths_and_labels()
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Training images: {len(splits['train'][0])}")
    print(f"Test images: {len(splits['test'][0])}")
    
    # Print class distribution
    train_labels = splits['train'][1]
    test_labels = splits['test'][1]
    print("\nClass Distribution:")
    print("Training set:")
    print(f"  Lie: {torch.sum(train_labels == 1).item()} ({(torch.sum(train_labels == 1).float() / len(train_labels) * 100).item():.1f}%)")
    print(f"  Truth: {torch.sum(train_labels == 0).item()} ({(torch.sum(train_labels == 0).float() / len(train_labels) * 100).item():.1f}%)")
    print("Test set:")
    print(f"  Lie: {torch.sum(test_labels == 1).item()} ({(torch.sum(test_labels == 1).float() / len(test_labels) * 100).item():.1f}%)")
    print(f"  Truth: {torch.sum(test_labels == 0).item()} ({(torch.sum(test_labels == 0).float() / len(test_labels) * 100).item():.1f}%)")
    
    # Calculate class weights for weighted sampling
    num_samples = len(train_labels)
    class_counts = torch.bincount(train_labels.long())
    class_weights = num_samples / (2 * class_counts)
    sample_weights = class_weights[train_labels.long()]
    
    # Save splits and weights to data_splits directory
    print("\nSaving splits and weights to files...")
    for split_name, (paths, labels) in splits.items():
        np.save(os.path.join('data_splits', f'{split_name}_paths.npy'), paths)
        torch.save(labels, os.path.join('data_splits', f'{split_name}_labels.pt'))
    
    # Save class weights for training
    torch.save(class_weights, os.path.join('data_splits', 'class_weights.pt'))
    torch.save(sample_weights, os.path.join('data_splits', 'sample_weights.pt'))
    
    print("\nDataset preparation completed!")
    print("Added class weights for handling class imbalance during training.")
    print("Using real-time face segmentation during data loading.")

if __name__ == "__main__":
    main() 