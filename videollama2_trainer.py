"""
VideoLLaMA 2 Trainer

This module provides training utilities for VideoLLaMA 2 models,
including data loading, training loops, and evaluation.

Based on the original implementation from:
https://github.com/DAMO-NLP-SG/VideoLLaMA-2
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import logging


class VideoLLaMA2Trainer:
    """Trainer class for VideoLLaMA 2 models."""
    
    def __init__(self,
                 model: nn.Module,
                 train_dataset,
                 val_dataset,
                 batch_size: int = 8,
                 learning_rate: float = 1e-5,
                 num_epochs: int = 10,
                 device: str = "cuda"):
        """
        Initialize the VideoLLaMA 2 trainer.
        
        Args:
            model: VideoLLaMA 2 model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            device: Device to train on (cuda/cpu)
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        
        # Setup data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and loss function
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Move model to device
        self.model.to(device)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            video_frames = batch['frames'].to(self.device)
            questions = batch['question']
            answers = batch['answer']
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(video_frames, questions)
            
            # Compute loss (simplified - actual implementation would be more complex)
            loss = self.criterion(outputs, answers)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == answers).sum().item()
            total_samples += answers.size(0)
            
            if batch_idx % 100 == 0:
                logging.info(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                video_frames = batch['frames'].to(self.device)
                questions = batch['question']
                answers = batch['answer']
                
                outputs = self.model(video_frames, questions)
                loss = self.criterion(outputs, answers)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == answers).sum().item()
                total_samples += answers.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self) -> Dict[str, List[float]]:
        """Run the full training process."""
        logging.info("Starting training...")
        
        for epoch in range(self.num_epochs):
            logging.info(f'Epoch {epoch+1}/{self.num_epochs}')
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            logging.info(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        torch.save(checkpoint, path)
        logging.info(f'Checkpoint saved to {path}')
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']
        logging.info(f'Checkpoint loaded from {path}')