import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from ..config import LOGS_DIR

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels (0 for Truth, 1 for Lie)
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_prob is not None and isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
        
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Add AUC if probabilities are provided
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = 0.5  # Default value for AUC
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plot and save confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save the plot
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['Truth', 'Lie'])
    plt.yticks([0.5, 1.5], ['Truth', 'Lie'])
    
    # Save plot if path is provided
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        # Save to default location
        os.makedirs(LOGS_DIR, exist_ok=True)
        plt.savefig(os.path.join(LOGS_DIR, f'{model_name}_confusion_matrix.png'))
        
    plt.close()
    
def plot_training_history(history, model_name, save_path=None):
    """Plot and save training history.
    
    Args:
        history: Dictionary containing loss and metrics history
        model_name: Name of the model
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        # Save to default location
        os.makedirs(LOGS_DIR, exist_ok=True)
        plt.savefig(os.path.join(LOGS_DIR, f'{model_name}_training_history.png'))
        
    plt.close() 