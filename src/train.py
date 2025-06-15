import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from .data.data_loader import get_train_val_loaders
from .models import create_custom_cnn, create_mobilenet, create_efficientnet, save_model
from .utils import calculate_metrics, plot_confusion_matrix, plot_training_history
from .config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    LEARNING_RATE,
    EPOCHS,
    CHECKPOINTS_DIR,
    MODELS_DIR,
    LOGS_DIR,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LR
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader, optimizer, criterion, model_name, 
                epochs=EPOCHS, checkpoint_dir=CHECKPOINTS_DIR):
    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Handle the case when val_loader is None - use test_loader as a fallback
    if val_loader is None:
        print("Warning: No validation loader provided. Using a portion of training data for validation.")
        # Create a temporary validation set from training data
        train_size = int(0.8 * len(train_loader.dataset))
        val_size = len(train_loader.dataset) - train_size
        train_subset, val_subset = random_split(
            train_loader.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create new train and validation loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=train_loader.batch_size,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory
        )
        
        print(f"Created temporary validation set with {val_size} samples")
    
    # Initialize variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    # Setup scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=REDUCE_LR_FACTOR, 
                                 patience=REDUCE_LR_PATIENCE, verbose=True, min_lr=MIN_LR)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Get timestamp for checkpoint naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Debug: calculate zero-coverage of masks in inputs to verify segmentation
        try:
            sample_inputs, _ = next(iter(train_loader))
            sample_inputs = sample_inputs.to(device)
            zero_ratio = (sample_inputs == 0).float().mean().item()
            print(f"[Debug] Mask zero-ratio in sample inputs: {zero_ratio:.2%}")
        except Exception:
            pass
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': loss.item(), 
                'acc': train_correct / train_total if train_total > 0 else 0
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_probabilities = []
        val_targets = []
        
        with torch.no_grad():
            # Progress bar for validation
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                
                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Collect predictions and targets for metrics
                val_predictions.extend(predicted.cpu().numpy())
                val_probabilities.extend(outputs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': loss.item(), 
                    'acc': val_correct / val_total if val_total > 0 else 0
                })
        
        # Calculate epoch statistics
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / train_total if train_total > 0 else 0
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Update scheduler
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)
        
        # Calculate additional metrics
        val_metrics = calculate_metrics(
            val_targets, val_predictions, val_probabilities
        )
        
        # Print epoch results
        time_elapsed = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{epochs} completed in {time_elapsed:.1f}s")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        print(f"Val Metrics: Precision={val_metrics['precision']:.4f}, "
              f"Recall={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}, "
              f"AUC={val_metrics.get('auc', 0):.4f}")
        print(f"Learning rate: {current_lr:.6f}")
        
        # Track best validation loss for early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            patience_counter = 0
            
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"{model_name}_{timestamp}_epoch{epoch+1}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss,
                'accuracy': epoch_val_acc,
                'metrics': val_metrics
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        print("-" * 80)
    
    # Plot training history
    plot_training_history(
        history, 
        model_name, 
        os.path.join(LOGS_DIR, f"{model_name}_{timestamp}_history.png")
    )
    
    # Return training history and metrics
    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'final_val_metrics': val_metrics
    }

def main(model_type='cnn', lr=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS, save_samples=False):
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Get training and validation data loaders
    print("Loading training and validation data...")
    train_loader, val_loader = get_train_val_loaders(
        batch_size=batch_size,
        use_cache=False,  
        save_sample_images=save_samples,  
        num_sample_images=20,  
        cache_dir='visualizations'  
    )
    
    # Create model
    print(f"Creating {model_type} model...")
    if model_type.lower() == 'cnn':
        model = create_custom_cnn()
        model_name = 'custom_cnn'
    elif model_type.lower() == 'mobilenet':
        model = create_mobilenet()
        model_name = 'mobilenet'
    elif model_type.lower() == 'efficientnet':
        model = create_efficientnet()
        model_name = 'efficientnet'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    print(f"Training {model_name} for {epochs} epochs...")
    train_results = train_model(
        model, train_loader, val_loader, optimizer, criterion, model_name, epochs
    )
    
    # Save the final model
    save_model(model, f"{model_name}_final")
    print(f"Saved final model to {os.path.join(MODELS_DIR, f'{model_name}_final.pt')}")
    
    # Print overall results
    print("\nTraining Summary:")
    print(f"Model: {model_name}")
    print(f"Best Validation Loss: {train_results['best_val_loss']:.4f}")
    print(f"Best Validation Accuracy: {train_results['best_val_acc']:.4f}")
    
    if save_samples:
        print(f"\nVisualization samples have been saved in the 'visualizations' directory")
    
    print("\nModel training completed!")

