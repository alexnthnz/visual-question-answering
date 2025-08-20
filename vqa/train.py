import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
from accelerate import Accelerator

from .model import VQAModel
from .data import _DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VQADataset(Dataset):
    """PyTorch Dataset for VQA training."""
    
    def __init__(self, dataset, model: VQAModel):
        self.dataset = dataset
        self.model = model
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            'image': sample['image'],
            'question': sample['question'],
            'answers': sample['answers']
        }


def collate_fn(batch, model: VQAModel):
    """Custom collate function for batching."""
    images = [item['image'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answers'] for item in batch]
    
    # Encode answers to targets
    targets = model.encode_answers(answers)
    
    return {
        'images': images,
        'questions': questions,
        'targets': targets
    }


def train(
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    data_fraction: float = 0.1,
    save_path: Optional[str] = None,
    vocab_path: Optional[str] = None
) -> VQAModel:
    """Train the VQA model with proper supervision."""
    
    # Initialize accelerator for distributed training
    accelerator = Accelerator()
    device = accelerator.device
    
    logger.info(f"Training on device: {device}")
    logger.info(f"Using data fraction: {data_fraction}")
    
    # Load dataset
    logger.info("Loading dataset...")
    train_split = f"train[:{int(data_fraction * 100)}%]"
    dataset = load_dataset("vqa", "vqa_v2", data_dir=str(_DATA_DIR), split=train_split)
    
    # Initialize model
    logger.info("Initializing model...")
    model = VQAModel()
    
    # Build answer vocabulary
    logger.info("Building answer vocabulary...")
    model.build_answer_vocab(dataset)
    
    # Save vocabulary if path provided
    if vocab_path:
        model.save_answer_vocab(vocab_path)
        logger.info(f"Saved vocabulary to {vocab_path}")
    
    # Create dataset and dataloader
    vqa_dataset = VQADataset(dataset, model)
    dataloader = DataLoader(
        vqa_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, model),
        num_workers=0  # Set to 0 for compatibility
    )
    
    # Initialize optimizer and loss
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()  # For soft targets
    
    # Prepare for accelerated training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Training loop
    model.train()
    total_steps = len(dataloader) * epochs
    progress_bar = tqdm(total=total_steps, desc="Training")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Forward pass
            logits = model(batch['images'], batch['questions'])
            targets = batch['targets'].to(device)
            
            # Compute loss
            loss = criterion(logits, targets)
            
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress
            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                'epoch': epoch + 1,
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{epoch_loss / num_batches:.4f}"
            })
        
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        if save_path and (epoch + 1) % 2 == 0:  # Save every 2 epochs
            checkpoint_path = f"{save_path}_epoch_{epoch + 1}.pt"
            accelerator.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    progress_bar.close()
    
    # Save final model
    if save_path:
        final_path = f"{save_path}_final.pt"
        accelerator.save(model.state_dict(), final_path)
        logger.info(f"Saved final model to {final_path}")
    
    return accelerator.unwrap_model(model)


def main():
    parser = argparse.ArgumentParser(description="Train VQA model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data-fraction", type=float, default=0.1, help="Fraction of dataset to use")
    parser.add_argument("--save-path", type=str, default="models/vqa_model", help="Path to save model")
    parser.add_argument("--vocab-path", type=str, default="models/answer_vocab.json", help="Path to save vocabulary")
    
    args = parser.parse_args()
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Train model
    trained_model = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_fraction=args.data_fraction,
        save_path=args.save_path,
        vocab_path=args.vocab_path
    )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
