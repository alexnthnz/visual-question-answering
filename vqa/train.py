import argparse

from datasets import load_dataset
from torch.optim import Adam
import torch
from tqdm import tqdm

from .model import VQAModel
from .data import _DATA_DIR


def train(epochs: int = 1) -> None:
    dataset = load_dataset("vqa", "vqa_v2", data_dir=str(_DATA_DIR), split="train[:1%]")
    model = VQAModel()
    optimizer = Adam(model.model.parameters(), lr=5e-6)

    for _ in range(epochs):
        for example in tqdm(dataset, desc="Training"):
            image = example["image"]
            question = example["question"]
            labels = torch.tensor([0])  # dummy target
            logits = model.forward(image, question)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    train(args.epochs)


if __name__ == "__main__":
    main()
