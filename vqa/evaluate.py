from datasets import load_dataset
from torchmetrics.functional import accuracy
import torch

from .model import VQAModel
from .data import _DATA_DIR


def evaluate() -> float:
    dataset = load_dataset("vqa", "vqa_v2", data_dir=str(_DATA_DIR), split="validation[:1%]")
    model = VQAModel()
    preds = []
    targets = []
    for example in dataset:
        image = example["image"]
        question = example["question"]
        logits = model.forward(image, question)
        preds.append(torch.argmax(logits, dim=1))
        targets.append(torch.tensor([0]))  # dummy target
    return accuracy(torch.cat(preds), torch.cat(targets)).item()


def main():
    acc = evaluate()
    print(f"Validation accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
