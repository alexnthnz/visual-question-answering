"""
Visual Question Answering (VQA) Package

A comprehensive VQA implementation using CLIP-based architecture with proper
answer classification, training, and evaluation capabilities.
"""

from .model import VQAModel
from .data import download_vqa, load_sample, show_sample
from .train import train
from .evaluate import evaluate, vqa_accuracy

__version__ = "1.0.0"
__author__ = "VQA Team"

__all__ = [
    "VQAModel",
    "download_vqa", 
    "load_sample", 
    "show_sample",
    "train",
    "evaluate",
    "vqa_accuracy"
]
