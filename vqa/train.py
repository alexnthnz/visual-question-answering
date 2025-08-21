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
from .config import ModelConfig, TrainingConfig

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
    model_config: ModelConfig = ModelConfig(),
    train_config: TrainingConfig = TrainingConfig()
) -> VQAModel:
    logger.info(f"Model Config: {model_config}")
    logger.info(f"Training Config: {train_config}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize accelerator for distributed training
    accelerator = Accelerator()
    device = accelerator.device
    
    logger.info(f"Training on device: {device}")
    logger.info(f"Using data fraction: {train_config.data_fraction}")
    
    # Load dataset
    logger.info("Loading dataset...")
    train_split = f"train[:{int(train_config.data_fraction * 100)}%]"
    dataset = load_dataset("vqa", "vqa_v2", data_dir=str(_DATA_DIR), split=train_split)
    
    # Initialize model with config
    logger.info("Initializing model...")
    model = VQAModel(
        model_name=model_config.model_name,
        num_answers=model_config.num_answers,
        hidden_dim=model_config.hidden_dim,
        dropout=model_config.dropout,
        unfreeze_clip=model_config.unfreeze_clip
    )
    
    # Build answer vocabulary
    logger.info("Building answer vocabulary...")
    model.build_answer_vocab(dataset)
    
    # Save vocabulary if path provided
    if train_config.vocab_path:
        model.save_answer_vocab(train_config.vocab_path)
        logger.info(f"Saved vocabulary to {train_config.vocab_path}")
    
    # Create dataset and dataloader
    vqa_dataset = VQADataset(dataset, model)
    dataloader = DataLoader(
        vqa_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, model),
        num_workers=0  # Set to 0 for compatibility
    )
    
    # Initialize optimizer and loss
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()  # For soft targets
    
    # Prepare for accelerated training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Training loop
    model.train()
    total_steps = len(dataloader) * train_config.epochs
    progress_bar = tqdm(total=total_steps, desc="Training")
    
    for epoch in range(train_config.epochs):
        epoch_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            with accelerator.autocast():
                logits = model(batch['images'], batch['questions'])
                targets = batch['targets'].to(device)
                
                loss = criterion(logits, targets) / train_config.gradient_accumulation_steps
                accumulated_loss += loss.item()
            
            accelerator.backward(loss)
            
            if (step + 1) % train_config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                # Log accumulated loss
                progress_bar.update(train_config.gradient_accumulation_steps)
                progress_bar.set_postfix({
                    'epoch': epoch + 1,
                    'loss': f"{accumulated_loss:.4f}",
                    'avg_loss': f"{epoch_loss / (num_batches + 1):.4f}"
                })
                epoch_loss += accumulated_loss
                accumulated_loss = 0.0
                num_batches += 1
        
        # Handle any remaining accumulation
        if accumulated_loss > 0:
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += accumulated_loss
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch + 1}/{train_config.epochs} - Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        if train_config.save_path and (epoch + 1) % train_config.save_every_n_epochs == 0:
            checkpoint_path = f"{train_config.save_path}_epoch_{epoch + 1}.pt"
            accelerator.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    progress_bar.close()
    
    # Save final model
    if train_config.save_path:
        final_path = f"{train_config.save_path}_final.pt"
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
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name")
    parser.add_argument("--num-answers", type=int, default=3000, help="Number of answers in vocabulary")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--unfreeze-clip", action="store_true", help="Unfreeze CLIP for fine-tuning")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--save-every-n-epochs", type=int, default=2, help="Save every N epochs")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Steps for gradient accumulation")
    
    args = parser.parse_args()
    
    # Create configs from args
    model_config = ModelConfig(
        model_name=args.model_name,
        num_answers=args.num_answers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        unfreeze_clip=args.unfreeze_clip
    )
    
    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_fraction=args.data_fraction,
        save_path=args.save_path,
        vocab_path=args.vocab_path,
        weight_decay=args.weight_decay,
        save_every_n_epochs=args.save_every_n_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Train model
    trained_model = train(
        model_config=model_config,
        train_config=train_config
    )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
