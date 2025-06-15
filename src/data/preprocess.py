"""
Face detection and preprocessing utilities.
"""

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image
from pathlib import Path

from ..config import (
    DATA_DIR, 
    TRAIN_DIR, 
    TEST_DIR, 
    IMAGE_SIZE, 
    FACE_MARGIN,
    TRAIN_CACHE_DIR,
    TEST_CACHE_DIR,
)

def detect_faces(image_paths, output_dir, target_size=IMAGE_SIZE, margin=FACE_MARGIN, batch_size=32):
    """Detect and save faces from a list of image paths.
    
    Args:
        image_paths: List of image paths
        output_dir: Directory to save face images
        target_size: Target size for face images
        margin: Margin around face for detection
        batch_size: Batch size for processing
        
    Returns:
        Dictionary mapping original paths to face detection results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize face detector
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = MTCNN(
        image_size=target_size[0],
        margin=margin,
        post_process=False,
        device=device
    )
    
    # Process images in batches
    results = {}
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Detecting faces"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        # Load images
        for img_path in batch_paths:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    batch_images.append(img)
                else:
                    batch_images.append(None)
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
                batch_images.append(None)
        
        # Detect faces
        try:
            batch_boxes, batch_probs, batch_landmarks = detector.detect(batch_images, landmarks=True)
        except:
            # If batch detection fails, process one by one
            batch_boxes, batch_probs, batch_landmarks = [], [], []
            for img in batch_images:
                try:
                    if img is not None:
                        boxes, probs, landmarks = detector.detect(img, landmarks=True)
                    else:
                        boxes, probs, landmarks = None, None, None
                except:
                    boxes, probs, landmarks = None, None, None
                batch_boxes.append(boxes)
                batch_probs.append(probs)
                batch_landmarks.append(landmarks)
        
        # Extract and save face tensors
        for j, img_path in enumerate(batch_paths):
            if j >= len(batch_images) or batch_images[j] is None:
                results[img_path] = None
                continue
                
            try:
                img_orig = batch_images[j]
                face_tensor = detector.extract(img_orig, None, None)
                
                if face_tensor is not None:
                    # Save face tensor
                    output_path = Path(output_dir) / f"{Path(img_path).stem}.pt"
                    torch.save(face_tensor, output_path)
                    results[img_path] = str(output_path)
                else:
                    results[img_path] = None
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                results[img_path] = None
    
    # Print summary
    success_count = sum(1 for v in results.values() if v is not None)
    print(f"Successfully processed {success_count}/{len(image_paths)} images ({success_count/len(image_paths)*100:.1f}%)")
    
    return results

def preprocess_dataset():
    """Preprocess the entire dataset, detecting faces and saving to cache directories."""
    # Ensure cache directories exist
    os.makedirs(TRAIN_CACHE_DIR, exist_ok=True)
    os.makedirs(TEST_CACHE_DIR, exist_ok=True)
    
    # Process training data
    print("Processing training data...")
    train_images = []
    for root, dirs, files in os.walk(TRAIN_DIR):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                train_images.append(os.path.join(root, file))
    
    print(f"Found {len(train_images)} training images")
    train_results = detect_faces(train_images, TRAIN_CACHE_DIR)
    
    # Process test data
    print("\nProcessing test data...")
    test_images = []
    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                test_images.append(os.path.join(root, file))
    
    print(f"Found {len(test_images)} test images")
    test_results = detect_faces(test_images, TEST_CACHE_DIR)
    
    # Return summary
    return {
        'train': {
            'total': len(train_images),
            'success': sum(1 for v in train_results.values() if v is not None)
        },
        'test': {
            'total': len(test_images),
            'success': sum(1 for v in test_results.values() if v is not None)
        }
    }

def main():
    """Main function to run face preprocessing."""
    results = preprocess_dataset()
    
    print("\nPreprocessing Summary:")
    print(f"Training images: {results['train']['success']}/{results['train']['total']} processed successfully")
    print(f"Test images: {results['test']['success']}/{results['test']['total']} processed successfully")

if __name__ == "__main__":
    main() 