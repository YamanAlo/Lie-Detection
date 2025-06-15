#!/usr/bin/env python
"""
Main entry point for the micro-expression detection project.
"""

import os
import argparse
from src.config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS,
    MODELS_DIR, CHECKPOINTS_DIR, DATA_SPLITS_DIR, LOGS_DIR
)

def setup_directories():
    """Set up the necessary directories for the project."""
    dirs = [
        MODELS_DIR, CHECKPOINTS_DIR, DATA_SPLITS_DIR, LOGS_DIR,
        'train_face_cache', 'test_face_cache', 'face_embeddings'
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Micro-expression Detection Project')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    
    # Prepare dataset command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare the dataset')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess faces and cache them')
    
    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Visualize dataset statistics')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', type=str, default='cnn', 
                              choices=['cnn', 'mobilenet', 'efficientnet'],
                              help='Model type to train')
    train_parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                              help='Learning rate')
    train_parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                              help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=EPOCHS,
                              help='Number of epochs to train for')
    train_parser.add_argument('--save_samples', action='store_true',
                              help='Save sample processed images during data loading')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test one or more trained models')
    test_parser.add_argument('--models', type=str, required=True, nargs='+',
                             help='Path(s) to the model file(s) to test')
    test_parser.add_argument('--model_types', type=str, required=True, nargs='+',
                             choices=['cnn', 'mobilenet', 'efficientnet'],
                             help='Type(s) of the model(s) to load')
    test_parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                             help='Batch size for testing')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the command
    if args.command == 'setup':
        setup_directories()
        
    elif args.command == 'prepare':
        # Import here to avoid circular imports
        from prepare_dataset import main as prepare_dataset
        prepare_dataset()
        
    elif args.command == 'preprocess':
        from src.data.preprocess import main as preprocess_faces
        preprocess_faces()
        
    elif args.command == 'visualize':
        from src.utils.visualization import main as visualize_data
        visualize_data()
        
    elif args.command == 'train':
        from src.train import main as train_model
        train_model(
            model_type=args.model,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            save_samples=args.save_samples
        )
        
    elif args.command == 'test':
        from src.test import main as test_model
        
        
        if len(args.models) != len(args.model_types):
            print("Error: Number of models and model types must match")
            return
            
        test_model(
            model_paths=args.models,
            model_types=args.model_types,
            batch_size=args.batch_size
        )
        
    else:
        parser.print_help()
        

if __name__ == "__main__":
    main() 