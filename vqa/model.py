from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import json
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from peft import LoraConfig, get_peft_model


class VQAModel(nn.Module):
    """VQA model with proper answer classification head."""
    
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-base-patch32",
        num_answers: int = 3000,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        unfreeze_clip: bool = False,
        fusion_type: str = "concat",
        num_fusion_layers: int = 2,
        num_attention_heads: int = 8,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        # Load CLIP components
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model = CLIPModel.from_pretrained(model_name)
        
        # Apply LoRA if enabled
        self.use_lora = use_lora
        if use_lora:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # Target attention layers in CLIP
                lora_dropout=lora_dropout,
                bias="none"
            )
            self.clip_model = get_peft_model(self.clip_model, lora_config)
            # Print trainable params for verification
            self.clip_model.print_trainable_parameters()
        
        # Get CLIP embedding dimension
        clip_dim = self.clip_model.config.projection_dim
        
        self.fusion_type = fusion_type
        
        if fusion_type == "cross_attention":
            # Cross-attention fusion using Transformer
            encoder_layer = TransformerEncoderLayer(
                d_model=clip_dim,
                nhead=num_attention_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="relu"
            )
            self.fusion_encoder = TransformerEncoder(encoder_layer, num_layers=num_fusion_layers)
        else:
            self.fusion_encoder = None
        
        # Answer classification head (input size depends on fusion)
        classifier_input_dim = clip_dim if fusion_type == "cross_attention" else clip_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
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
        
        # Handle CLIP freezing (for non-LoRA case)
        if not use_lora:
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
        
        # Get CLIP embeddings - use last_hidden_state for finer features
        clip_outputs = self.clip_model(**inputs, output_hidden_states=True)
        
        # Extract features (use last hidden state for text and vision)
        # For vision: [batch, num_patches + 1, dim] (including CLS)
        # For text: [batch, seq_len, dim]
        image_features = clip_outputs.vision_model_output.last_hidden_state  # [batch, patches+1, dim]
        text_features = clip_outputs.text_model_output.last_hidden_state     # [batch, seq_len, dim]
        
        if self.fusion_type == "cross_attention":
            # Prepare for cross-attention: treat text as query, image as key/value
            # Average text features to get a single vector per batch item
            text_query = text_features.mean(dim=1).unsqueeze(1)  # [batch, 1, dim]
            
            # Concatenate text query with image features
            combined = torch.cat([text_query, image_features], dim=1)  # [batch, 1 + patches+1, dim]
            
            # Apply fusion encoder
            fused_features = self.fusion_encoder(combined)  # [batch, seq_len, dim]
            
            # Pool the fused features (global average pooling)
            pooled_features = fused_features.mean(dim=1)  # [batch, dim]
        else:
            # Original concat method using pooled embeds
            image_pooled = clip_outputs.image_embeds
            text_pooled = clip_outputs.text_embeds
            pooled_features = torch.cat([image_pooled, text_pooled], dim=1)
        
        # Classify answers
        logits = self.classifier(pooled_features)
        
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
        """Unfreeze CLIP parameters for fine-tuning. If using LoRA, this enables full fine-tuning post-LoRA."""
        if self.use_lora:
            # Merge LoRA adapters and unfreeze
            self.clip_model = self.clip_model.merge_and_unload()
            self.use_lora = False  # Disable LoRA after merging
        for param in self.clip_model.parameters():
            param.requires_grad = True
