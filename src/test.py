"""
Test script for evaluating trained models on the test dataset.
This script focuses solely on model evaluation, not training.
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import argparse
from datetime import datetime

from .data import get_test_loader
from .models import load_model
from .utils import calculate_metrics, plot_confusion_matrix
from .config import BATCH_SIZE, IMAGE_SIZE, LOGS_DIR, MODELS_DIR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model(model, test_loader, criterion, model_name):
    """Test the model on the test dataset.
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        criterion: The loss function
        model_name: Name of the model (for logging)
        
    Returns:
        Dictionary with test metrics
    """
    model.eval()
    test_loss = 0.0
    test_predictions = []
    test_probabilities = []
    test_targets = []
    
    # Evaluate on test data
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            
            # Collect predictions and targets for metrics
            test_predictions.extend(predicted.cpu().numpy())
            test_probabilities.extend(outputs.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
    
    
    test_loss = test_loss / len(test_loader.dataset)
    test_metrics = calculate_metrics(
        test_targets, test_predictions, test_probabilities
    )
    
    
    print(f"\n=== {model_name} Test Results ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    if 'auc' in test_metrics:
        print(f"Test AUC: {test_metrics['auc']:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    target_names = ['Truth', 'Lie']
    report = classification_report(
        test_targets, test_predictions, target_names=target_names
    )
    print(report)
    
    return {
        'name': model_name,
        'loss': test_loss,
        'metrics': test_metrics,
        'predictions': test_predictions,
        'probabilities': test_probabilities,
        'targets': test_targets
    }

def plot_roc_curve(results, timestamp=None):
    """Plot ROC curve for one or multiple models.
    
    Args:
        results: List of dictionaries with test results
        timestamp: Optional timestamp for file naming
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC for each model
    for result in results:
        model_name = result['name']
        y_true = result['targets']
        y_prob = result['probabilities']
        
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    
   
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    plt.savefig(os.path.join(LOGS_DIR, f'roc_curve_comparison_{timestamp}.png'))
    plt.close()
    
    print(f"Saved ROC curve comparison to {os.path.join(LOGS_DIR, f'roc_curve_comparison_{timestamp}.png')}")

def compare_models(results, timestamp=None):
    """Compare multiple models' performance.
    
    Args:
        results: List of dictionaries with test results
        timestamp: Optional timestamp for file naming
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create comparison table
    print("\n=== Model Comparison ===")
    print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'AUC':<10}")
    print("-" * 65)
    
    # Initialize data for plotting
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    aucs = []
    
    for result in results:
        model_name = result['name']
        metrics = result['metrics']
        
        # Print metrics
        print(f"{model_name:<15} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} "
              f"{metrics['f1']:<10.4f} "
              f"{metrics.get('auc', 0):<10.4f}")
        
        # Collect data for plotting
        model_names.append(model_name)
        accuracies.append(metrics['accuracy'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1'])
        aucs.append(metrics.get('auc', 0))
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    x = np.arange(len(model_names))
    width = 0.15
    
    plt.bar(x - 2*width, accuracies, width, label='Accuracy')
    plt.bar(x - width, precisions, width, label='Precision')
    plt.bar(x, recalls, width, label='Recall')
    plt.bar(x + width, f1_scores, width, label='F1 Score')
    plt.bar(x + 2*width, aucs, width, label='AUC')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(LOGS_DIR, exist_ok=True)
    plt.savefig(os.path.join(LOGS_DIR, f'model_comparison_{timestamp}.png'))
    plt.close()
    
    print(f"Saved model comparison chart to {os.path.join(LOGS_DIR, f'model_comparison_{timestamp}.png')}")

def main(model_paths, model_types, batch_size=BATCH_SIZE):
    """Main function to test one or multiple trained models.
    
    Args:
        model_paths: List of paths to model files
        model_types: List of model types ('cnn', 'mobilenet', 'efficientnet')
        batch_size: Batch size for testing
    """
    if isinstance(model_paths, str):
        model_paths = [model_paths]
    if isinstance(model_types, str):
        model_types = [model_types] * len(model_paths)
    
    if len(model_paths) != len(model_types):
        raise ValueError("Number of model paths and model types must match")
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Get only the test data loader
    print("Loading test data...")
    # Disable caching for testing to avoid inconsistencies
    test_loader = get_test_loader(batch_size=batch_size, use_cache=False)
    
    # Define loss function
    criterion = nn.BCELoss()
    
    # Get timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Test each model
    results = []
    for i, (model_path, model_type) in enumerate(zip(model_paths, model_types)):
       
        if os.path.isfile(model_path):
            normalized_path = model_path
        else:
            model_name = model_path.split('.')[0] 
            normalized_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
        
        # Load model
        print(f"\n[{i+1}/{len(model_paths)}] Loading model from {normalized_path}...")
        model = load_model(normalized_path)
        model_name = os.path.splitext(os.path.basename(normalized_path))[0]
        
        # Move model to device
        model = model.to(device)
        
        # Test the model
        result = test_model(model, test_loader, criterion, model_name)
        results.append(result)
        
        # Plot confusion matrix for this model
        plot_confusion_matrix(
            result['targets'], 
            result['predictions'], 
            model_name, 
            os.path.join(LOGS_DIR, f"{model_name}_confusion_{timestamp}.png")
        )
    
    # Compare models if multiple
    if len(results) > 1:
        compare_models(results, timestamp)
    
    # Plot ROC curves
    plot_roc_curve(results, timestamp)
    
    print(f"\nTest evaluation completed! Results saved to {LOGS_DIR}")
    
    return results

