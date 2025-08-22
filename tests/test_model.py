import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import torch
from vqa.model import VQAModel

@pytest.fixture
def model():
    return VQAModel(num_answers=10)

def test_model_init(model):
    assert model.answer_vocab is None
    assert model.idx_to_answer is None

def test_forward_shape(model):
    images = [torch.rand(3, 224, 224)]  # Dummy image
    questions = ["What is this?"]
    logits = model(images, questions)
    assert logits.shape == (1, 10)  # batch_size, num_answers

def test_unfreeze_clip():
    model = VQAModel(unfreeze_clip=True)
    assert all(param.requires_grad for param in model.clip_model.parameters())

def test_lora_integration():
    model = VQAModel(use_lora=True, lora_rank=8)
    assert model.use_lora is True
    # Verify some parameters are adapted
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0  # LoRA adds trainable params
    model.unfreeze_clip()
    assert all(p.requires_grad for p in model.clip_model.parameters())  # Full unfreeze after LoRA merge

def test_build_vocab(model):
    mock_dataset = [{"answers": [{"answer": "yes"}, {"answer": "no"}]}]
    vocab = model.build_answer_vocab(mock_dataset)
    assert "yes" in vocab
    assert "no" in vocab

# Add more tests as needed
