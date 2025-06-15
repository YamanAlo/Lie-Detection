import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import math
import glob
from collections import defaultdict
import numpy as np
import torch

from ..config import (
    DATA_DIR, 
    TRAIN_DIR, 
    TEST_DIR, 
    DATA_SPLITS_DIR,
    LOGS_DIR
)

def find_image_paths():
    """Find all image paths in the dataset."""
    image_paths = []
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    
    for split_name, split_path in [('Train', TRAIN_DIR), ('Test', TEST_DIR)]:
        if not os.path.isdir(split_path):
            print(f"Warning: {split_path} not found")
            continue
            
        for label in ['Lie', 'Truth']:
            label_path = os.path.join(split_path, label)
            if not os.path.isdir(label_path):
                continue
                
            participants = [d for d in os.listdir(label_path) 
                          if os.path.isdir(os.path.join(label_path, d))]
            
            for participant in participants:
                participant_path = os.path.join(label_path, participant)
                questions = [q for q in os.listdir(participant_path) 
                           if os.path.isdir(os.path.join(participant_path, q))]
                
                for question in questions:
                    question_path = os.path.join(participant_path, question)
                    for ext in image_extensions:
                        files = glob.glob(os.path.join(question_path, ext))
                        for f_path in files:
                            image_paths.append((
                                os.path.normpath(f_path),
                                split_name,
                                label,
                                participant,
                                question
                            ))
    
    return image_paths

def visualize_samples(image_paths, num_samples=8, save_path=None):
    """Visualize random sample images from the dataset."""
    if not image_paths:
        print("No images found to visualize!")
        return
        
    # Select random samples
    num_samples = min(num_samples, len(image_paths))
    random_samples = random.sample(image_paths, num_samples)
    
    # Setup the plot grid
    cols = min(4, num_samples)
    rows = math.ceil(num_samples / cols)
    fig = plt.figure(figsize=(15, 4 * rows))
    
    # Plot each sample
    for idx, (img_path, split, label, participant, question) in enumerate(random_samples):
        try:
            # Create subplot
            ax = fig.add_subplot(rows, cols, idx + 1)
            
            # Load and display image
            img = Image.open(img_path)
            ax.imshow(img)
            
            # Add title with metadata
            title = f"{split}/{label}\n{participant}\n{question[:30]}..."
            ax.set_title(title, fontsize=8)
            ax.axis('off')
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            ax.text(0.5, 0.5, f'Error loading\n{os.path.basename(img_path)}',
                   ha='center', va='center', color='red')
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = os.path.join(LOGS_DIR, 'sample_images.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Sample images saved to {save_path}")

def analyze_data_distribution(save_path=None):
    """Analyze the distribution of labels in the dataset using saved paths and labels."""
    try:
        train_paths = np.load(os.path.join(DATA_SPLITS_DIR, 'train_paths.npy'), allow_pickle=True)
        train_labels = torch.load(os.path.join(DATA_SPLITS_DIR, 'train_labels.pt'))
        test_paths = np.load(os.path.join(DATA_SPLITS_DIR, 'test_paths.npy'), allow_pickle=True)
        test_labels = torch.load(os.path.join(DATA_SPLITS_DIR, 'test_labels.pt'))
        
        # Convert to numpy for easier processing
        if isinstance(train_labels, torch.Tensor):
            train_labels = train_labels.numpy()
        if isinstance(test_labels, torch.Tensor):
            test_labels = test_labels.numpy()
        
        print("\nDataset Distribution:")
        print(f"Train set: {len(train_paths)} samples")
        print(f"  Truth: {np.sum(train_labels == 0)} ({100 * np.sum(train_labels == 0) / len(train_labels):.1f}%)")
        print(f"  Lie: {np.sum(train_labels == 1)} ({100 * np.sum(train_labels == 1) / len(train_labels):.1f}%)")
        
        print(f"Test set: {len(test_paths)} samples")
        print(f"  Truth: {np.sum(test_labels == 0)} ({100 * np.sum(test_labels == 0) / len(test_labels):.1f}%)")
        print(f"  Lie: {np.sum(test_labels == 1)} ({100 * np.sum(test_labels == 1) / len(test_labels):.1f}%)")
        
        # Visualize the distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Train set
        train_counts = [np.sum(train_labels == 0), np.sum(train_labels == 1)]
        ax1.bar(['Truth', 'Lie'], train_counts, color=['green', 'red'])
        ax1.set_title('Train Set Distribution')
        ax1.set_ylabel('Number of Samples')
        for i, count in enumerate(train_counts):
            ax1.text(i, count + 10, str(count), ha='center')
        
        # Test set
        test_counts = [np.sum(test_labels == 0), np.sum(test_labels == 1)]
        ax2.bar(['Truth', 'Lie'], test_counts, color=['green', 'red'])
        ax2.set_title('Test Set Distribution')
        ax2.set_ylabel('Number of Samples')
        for i, count in enumerate(test_counts):
            ax2.text(i, count + 5, str(count), ha='center')
        
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = os.path.join(LOGS_DIR, 'label_distribution.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
        print(f"Label distribution plot saved to {save_path}")
        return True
        
    except (FileNotFoundError, IOError) as e:
        print(f"\nCouldn't analyze data distribution: {e}")
        return False

def print_dataset_stats(image_paths):
    """Print statistics about the dataset."""
    if not image_paths:
        print("No images found!")
        return
        
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # Collect statistics
    for path, split, label, participant, _ in image_paths:
        stats[split][label][participant] += 1
    
    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    
    total_images = 0
    for split in stats:
        split_total = 0
        print(f"\n{split}:")
        
        for label in stats[split]:
            label_total = 0
            print(f"  {label}:")
            
            for participant in sorted(stats[split][label].keys()):
                count = stats[split][label][participant]
                print(f"    {participant}: {count}")
                label_total += count
                
            print(f"    --- {label} Total: {label_total} ---")
            split_total += label_total
            
        print(f"  --- {split} Total: {split_total} ---")
        total_images += split_total
    
    print(f"\nTotal Images: {total_images}")

def main():
    """Main function to visualize dataset information."""
    # Ensure logs directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    print("Finding image paths...")
    image_paths = find_image_paths()
    
    if not image_paths:
        print("No images found in the dataset!")
        return
        
    print(f"Found {len(image_paths)} images")
    
    # Print dataset statistics
    print_dataset_stats(image_paths)
    
    # Analyze the label distribution from saved files
    analyze_data_distribution()
    
    # Visualize random samples
    print("\nVisualizing random samples...")
    visualize_samples(image_paths)

if __name__ == "__main__":
    main() 