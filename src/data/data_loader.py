import os
import numpy as np
import cv2
from facenet_pytorch import MTCNN
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import hashlib
from ..config import (
    DATA_SPLITS_DIR,
    TRAIN_DIR,
    TEST_DIR,
    TRAIN_CACHE_DIR,
    TEST_CACHE_DIR,
    BATCH_SIZE,
    IMAGE_SIZE,
    FACE_MARGIN,
    VALIDATION_SPLIT,
    USE_SEGMENTATION,
    TRAIN_MASK_DIR,
    TEST_MASK_DIR
)
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
from ..utils.face_segmenter import FaceMeshSegmenter

class MicroExpressionDataset(Dataset):
    """Dataset for loading and preprocessing micro-expression images.
    
    This class handles:
    1. Loading images with face detection
    2. Creating a cache of detected faces to disk for faster loading
    3. Providing labeled data for training
    """
    def __init__(self, image_paths, labels, target_size=IMAGE_SIZE, cache_dir='face_cache', use_cache=True, save_sample_images=False, num_sample_images=10):
        self.image_paths = image_paths
        self.labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.float32)
        self.target_size = target_size
        self.use_cache = use_cache
        self.save_sample_images = save_sample_images
        self.num_sample_images = num_sample_images
        self._saved_samples_count = 0
        
        # Set up cache directory if needed for caching or saving samples
        if use_cache or save_sample_images:
            self.cache_dir = Path(cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize face detector and segmenter
        self.detector = MTCNN(
            image_size=target_size[0],
            margin=FACE_MARGIN,
            post_process=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        if USE_SEGMENTATION:
            self.segmenter = FaceMeshSegmenter()
        
        # Filter valid images and create cache or save samples
        self.valid_indices = []
        self._prepare_dataset()
        
        print(f"Dataset ready with {len(self.valid_indices)} valid face images")
        
    def _prepare_dataset(self):
        """Prepare dataset by detecting faces and caching results"""
        print("Preparing dataset...")
        for idx, (img_path, label) in enumerate(tqdm(zip(self.image_paths, self.labels), total=len(self.image_paths))):
            if self.use_cache:
                # Generate a unique filename based on the image path to avoid collisions
                hash_name = hashlib.md5(str(img_path).encode()).hexdigest()
                cache_path = self.cache_dir / f"{Path(img_path).stem}_{hash_name}.pt"
                
                if cache_path.exists():
                    # If cached result exists and is valid, add to valid indices
                    try:
                        face_tensor = torch.load(cache_path)
                        if face_tensor is not None:
                            self.valid_indices.append(idx)
                    except:
                        pass
                else:
                    # Process image and cache result
                    face_tensor = self._process_image(img_path)
                    if face_tensor is not None:
                        torch.save(face_tensor, cache_path)
                        self.valid_indices.append(idx)
                        # Save sample image if requested
                        if self.save_sample_images and self._saved_samples_count < self.num_sample_images:
                            sample_img_path = self.cache_dir / f"{Path(img_path).stem}_{hash_name}.png"
                            save_image(face_tensor.cpu(), sample_img_path, normalize=True)
                            self._saved_samples_count += 1
            else:
                # If not using cache, just check if face can be detected
                face_tensor = self._process_image(img_path)
                if face_tensor is not None:
                    self.valid_indices.append(idx)
                    # Save sample image if requested (even without caching)
                    if self.save_sample_images and self._saved_samples_count < self.num_sample_images:
                        hash_name = hashlib.md5(str(img_path).encode()).hexdigest()
                        sample_img_path = Path(self.cache_dir) / f"{Path(img_path).stem}_{hash_name}.png"
                        save_image(face_tensor.cpu(), sample_img_path, normalize=True)
                        self._saved_samples_count += 1
                        print(f"Saved visualization sample: {sample_img_path}")
    
    def _process_image(self, image_path):
        """Process single image: detect face, align, and normalize."""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            # Convert to RGB and PIL Image
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            
            # Detect and align face
            face_tensor = self.detector(img_pil)
            if face_tensor is None:
                return None
            face_tensor = face_tensor.float()
            
            # Apply segmentation masks if enabled
            if USE_SEGMENTATION:
                # Get region masks using MediaPipe
                region_masks = self.segmenter.get_region_masks(
                    cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR),
                    target_size=(face_tensor.shape[2], face_tensor.shape[1])
                )
                
                # Apply masks - focus on eyes and lips for micro-expressions
                combined_mask = None
                for region in ['left_eye', 'right_eye', 'lips']:
                    if region in region_masks:
                        mask = region_masks[region].to(face_tensor.device)
                        if combined_mask is None:
                            combined_mask = mask
                        else:
                            combined_mask = torch.max(combined_mask, mask)
                
                if combined_mask is not None:
                    # Apply combined mask to face tensor
                    face_tensor = face_tensor * combined_mask
             
            return face_tensor
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        true_idx = self.valid_indices[idx]
        img_path = self.image_paths[true_idx]
        label = self.labels[true_idx]
        
        if self.use_cache:
            # Load cached face tensor
            hash_name = hashlib.md5(str(img_path).encode()).hexdigest()
            cache_path = self.cache_dir / f"{Path(img_path).stem}_{hash_name}.pt"
            face_tensor = torch.load(cache_path)
        else:
            # Process image on-the-fly
            face_tensor = self._process_image(img_path)
        
        # Ensure tensor dtype matches model (float32)
        face_tensor = face_tensor.float()
        
        return face_tensor, label

def get_data_loaders(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, use_cache=False,
                      num_workers=2, validation_split=VALIDATION_SPLIT,
                      save_sample_images=False, num_sample_images=10):
    """Get train, validation, and test DataLoaders by delegating to specialized loaders."""
    # Get train and validation loaders
    train_loader, val_loader = get_train_val_loaders(
        batch_size=batch_size,
        image_size=image_size,
        use_cache=use_cache,
        num_workers=num_workers,
        validation_split=validation_split,
        save_sample_images=save_sample_images,
        num_sample_images=num_sample_images
    )
    # Get test loader
    test_loader = get_test_loader(
        batch_size=batch_size,
        image_size=image_size,
        use_cache=use_cache,
        num_workers=num_workers,
        save_sample_images=save_sample_images,
        num_sample_images=num_sample_images
    )
    return train_loader, val_loader, test_loader

# Add helper to load only the test split without preprocessing train/val
def get_test_loader(batch_size=BATCH_SIZE,
                     image_size=IMAGE_SIZE,
                     use_cache=False,
                     num_workers=0,
                     save_sample_images=False,
                     num_sample_images=10):
    """Get only the test DataLoader without processing train or validation splits."""
    # Load test paths and labels
    test_paths = np.load(os.path.join(DATA_SPLITS_DIR, 'test_paths.npy'), allow_pickle=True)
    test_labels = torch.load(os.path.join(DATA_SPLITS_DIR, 'test_labels.pt'))

    # Create test dataset (face detection & caching as usual)
    test_dataset = MicroExpressionDataset(
        test_paths, test_labels, image_size,
        cache_dir=TEST_CACHE_DIR, use_cache=use_cache,
        save_sample_images=save_sample_images, num_sample_images=num_sample_images
    )

    # Create and return DataLoader for test
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return test_loader

# Helper to load only train and validation DataLoaders without processing test data
def get_train_val_loaders(batch_size=BATCH_SIZE,
                         image_size=IMAGE_SIZE,
                         use_cache=False,
                         num_workers=2,
                         validation_split=VALIDATION_SPLIT,
                         save_sample_images=False,
                         num_sample_images=10,
                         cache_dir=None):
    """Get DataLoaders for training and validation only, skipping test data."""
    if not use_cache:
        num_workers = 0

    # Load train paths and labels
    train_paths = np.load(os.path.join(DATA_SPLITS_DIR, 'train_paths.npy'), allow_pickle=True)
    train_labels = torch.load(os.path.join(DATA_SPLITS_DIR, 'train_labels.pt'))

    # Check for saved validation split
    val_paths_file = os.path.join(DATA_SPLITS_DIR, 'val_paths.npy')
    val_labels_file = os.path.join(DATA_SPLITS_DIR, 'val_labels.pt')
    has_val_split = os.path.exists(val_paths_file) and os.path.exists(val_labels_file)
    if has_val_split:
        val_paths = np.load(val_paths_file, allow_pickle=True)
        val_labels = torch.load(val_labels_file)
        print(f"Loaded existing validation split with {len(val_paths)} samples")

    # Print dataset size
    print(f"Loaded {len(train_paths)} training samples")
    if has_val_split:
        print(f"Validation labels distribution: Truth={torch.sum(val_labels == 0).item()}, Lie={torch.sum(val_labels == 1).item()}")

    # Build datasets
    if has_val_split:
        train_dataset = MicroExpressionDataset(
            train_paths, train_labels, image_size,
            cache_dir=cache_dir if cache_dir is not None else TRAIN_CACHE_DIR, use_cache=use_cache,
            save_sample_images=save_sample_images, num_sample_images=num_sample_images
        )
        val_dataset = MicroExpressionDataset(
            val_paths, val_labels, image_size,
            cache_dir=cache_dir if cache_dir is not None else TRAIN_CACHE_DIR, use_cache=use_cache,
            save_sample_images=False, num_sample_images=num_sample_images
        )
    elif validation_split > 0:
        full_dataset = MicroExpressionDataset(
            train_paths, train_labels, image_size,
            cache_dir=cache_dir if cache_dir is not None else TRAIN_CACHE_DIR, use_cache=use_cache,
            save_sample_images=save_sample_images, num_sample_images=num_sample_images
        )
        val_size = int(len(full_dataset) * validation_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"Created temporary validation split with {val_size} samples")
    else:
        train_dataset = MicroExpressionDataset(
            train_paths, train_labels, image_size,
            cache_dir=cache_dir if cache_dir is not None else TRAIN_CACHE_DIR, use_cache=use_cache,
            save_sample_images=save_sample_images, num_sample_images=num_sample_images
        )
        val_dataset = None
        print("No validation split created (validation_split=0)")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    ) if val_dataset else None
    return train_loader, val_loader 