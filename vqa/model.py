from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import json


class VQAModel(nn.Module):
    """VQA model with proper answer classification head."""
    
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-base-patch32",
        num_answers: int = 3000,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        unfreeze_clip: bool = False
    ) -> None:
        super().__init__()
        
        # Load CLIP components
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model = CLIPModel.from_pretrained(model_name)
        
        # Get CLIP embedding dimension
        clip_dim = self.clip_model.config.projection_dim
        
        # Answer classification head
        self.classifier = nn.Sequential(
            nn.Linear(clip_dim * 2, hidden_dim),  # Concat image + text features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_answers)
        )
        
        # Answer vocabulary
        self.answer_vocab: Optional[Dict[str, int]] = None
        self.idx_to_answer: Optional[Dict[int, str]] = None
        
        # Handle CLIP freezing
        for param in self.clip_model.parameters():
            param.requires_grad = unfreeze_clip
    
    def build_answer_vocab(self, dataset) -> Dict[str, int]:
        """Build answer vocabulary from dataset."""
        answer_counts = {}
        
        # Count all answers
        for sample in dataset:
            for answer_obj in sample["answers"]:
                answer = answer_obj["answer"].lower().strip()
                answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        # Select top answers
        sorted_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
        top_answers = [answer for answer, _ in sorted_answers[:self.classifier[-1].out_features]]
        
        # Create vocab mappings
        self.answer_vocab = {answer: idx for idx, answer in enumerate(top_answers)}
        self.idx_to_answer = {idx: answer for answer, idx in self.answer_vocab.items()}
        
        return self.answer_vocab
    
    def save_answer_vocab(self, path: Union[str, Path]) -> None:
        """Save answer vocabulary to file."""
        if self.answer_vocab is None:
            raise ValueError("Answer vocabulary not built yet")
        
        vocab_data = {
            "answer_vocab": self.answer_vocab,
            "idx_to_answer": self.idx_to_answer
        }
        
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    def load_answer_vocab(self, path: Union[str, Path]) -> None:
        """Load answer vocabulary from file."""
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        self.answer_vocab = vocab_data["answer_vocab"]
        self.idx_to_answer = {int(k): v for k, v in vocab_data["idx_to_answer"].items()}
    
    def encode_answers(self, answers: List[List[Dict]]) -> torch.Tensor:
        """Encode answers to target labels for training."""
        if self.answer_vocab is None:
            raise ValueError("Answer vocabulary not built yet")
        
        batch_size = len(answers)
        num_classes = len(self.answer_vocab)
        
        # Create soft targets (multiple annotators per question)
        targets = torch.zeros(batch_size, num_classes)
        
        for i, answer_list in enumerate(answers):
            answer_counts = {}
            total_answers = len(answer_list)
            
            # Count each answer
            for answer_obj in answer_list:
                answer = answer_obj["answer"].lower().strip()
                if answer in self.answer_vocab:
                    answer_counts[answer] = answer_counts.get(answer, 0) + 1
            
            # Convert to soft targets
            for answer, count in answer_counts.items():
                idx = self.answer_vocab[answer]
                targets[i, idx] = count / total_answers
        
        return targets
    
    def forward(
        self, 
        images: Union[torch.Tensor, List], 
        questions: List[str]
    ) -> torch.Tensor:
        """Forward pass through the model."""
        # Process inputs through CLIP
        inputs = self.processor(
            text=questions, 
            images=images, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move to same device as model
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get CLIP embeddings
        clip_outputs = self.clip_model(**inputs)
        
        # Extract normalized features
        image_features = clip_outputs.image_embeds  # [batch_size, clip_dim]
        text_features = clip_outputs.text_embeds    # [batch_size, clip_dim]
        
        # Concatenate features
        combined_features = torch.cat([image_features, text_features], dim=1)
        
        # Classify answers
        logits = self.classifier(combined_features)
        
        return logits
    
    def predict(self, images, questions: List[str], top_k: int = 1) -> List[List[Tuple[str, float]]]:
        """Generate predictions with confidence scores."""
        if self.idx_to_answer is None:
            raise ValueError("Answer vocabulary not loaded")
        
        self.eval()
        with torch.no_grad():
            logits = self.forward(images, questions)
            probs = torch.softmax(logits, dim=-1)
            
            predictions = []
            for i in range(len(questions)):
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probs[i], top_k)
                
                question_preds = []
                for prob, idx in zip(top_probs, top_indices):
                    answer = self.idx_to_answer[idx.item()]
                    confidence = prob.item()
                    question_preds.append((answer, confidence))
                
                predictions.append(question_preds)
        
        return predictions
    
    def unfreeze_clip(self) -> None:
        """Unfreeze CLIP parameters for fine-tuning."""
        for param in self.clip_model.parameters():
            param.requires_grad = True
