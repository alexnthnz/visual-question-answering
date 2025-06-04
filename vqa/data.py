import argparse
from pathlib import Path
from typing import Tuple

from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

_DATA_DIR = Path("data")


def download_vqa() -> None:
    """Download the VQA v2.0 dataset using the HuggingFace datasets library."""
    load_dataset("vqa", "vqa_v2", data_dir=str(_DATA_DIR))


def load_sample() -> Tuple[Image.Image, str, str]:
    """Return a sample image, question, and answer."""
    ds = load_dataset("vqa", "vqa_v2", data_dir=str(_DATA_DIR), split="validation")
    sample = ds[0]
    image = Image.open(sample["image"]).convert("RGB")
    question = sample["question"]
    answer = sample["answers"]["text"][0]
    return image, question, answer


def show_sample() -> None:
    image, question, answer = load_sample()
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Q: {question}\nA: {answer}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download VQA v2.0")
    parser.add_argument("--show-sample", action="store_true", help="Show a sample")
    args = parser.parse_args()

    if args.download:
        download_vqa()
    if args.show_sample:
        show_sample()


if __name__ == "__main__":
    main()
