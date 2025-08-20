import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional

from datasets import load_dataset, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DATA_DIR = Path("data")


def download_vqa() -> None:
    """Download the VQA v2.0 dataset using the HuggingFace datasets library."""
    logger.info("Downloading VQA v2.0 dataset...")
    
    try:
        # Download both train and validation splits
        train_dataset = load_dataset("vqa", "vqa_v2", data_dir=str(_DATA_DIR), split="train")
        val_dataset = load_dataset("vqa", "vqa_v2", data_dir=str(_DATA_DIR), split="validation")
        
        logger.info(f"Downloaded {len(train_dataset)} training samples")
        logger.info(f"Downloaded {len(val_dataset)} validation samples")
        logger.info(f"Data saved to {_DATA_DIR}")
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


def load_sample(split: str = "validation", index: int = 0) -> Tuple[Image.Image, str, List[str]]:
    """
    Return a sample image, question, and answers.
    
    Args:
        split: Dataset split to use ('train' or 'validation')
        index: Index of the sample to load
    
    Returns:
        Tuple of (image, question, list of answers)
    """
    try:
        ds = load_dataset("vqa", "vqa_v2", data_dir=str(_DATA_DIR), split=split)
        sample = ds[index]
        
        # Load and convert image
        if isinstance(sample["image"], str):
            image = Image.open(sample["image"]).convert("RGB")
        else:
            image = sample["image"].convert("RGB")
        
        question = sample["question"]
        
        # Extract all answers (VQA has multiple annotators)
        if isinstance(sample["answers"], dict):
            answers = sample["answers"]["answer"]
        else:
            answers = [ans["answer"] for ans in sample["answers"]]
        
        return image, question, answers
        
    except Exception as e:
        logger.error(f"Error loading sample: {e}")
        raise


def show_sample(split: str = "validation", index: int = 0, save_path: Optional[str] = None) -> None:
    """
    Display a sample from the dataset.
    
    Args:
        split: Dataset split to use
        index: Index of the sample to display
        save_path: Optional path to save the visualization
    """
    try:
        image, question, answers = load_sample(split, index)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis("off")
        
        # Format answers
        answer_text = ", ".join(set(answers))  # Remove duplicates
        answer_counts = {ans: answers.count(ans) for ans in set(answers)}
        most_common = max(answer_counts, key=answer_counts.get)
        
        title = f"Question: {question}\n"
        title += f"Most Common Answer: {most_common} ({answer_counts[most_common]}/10)\n"
        title += f"All Answers: {answer_text}"
        
        plt.title(title, fontsize=12, wrap=True, pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error showing sample: {e}")
        raise


def analyze_dataset(split: str = "train", max_samples: Optional[int] = None) -> Dict:
    """
    Analyze the VQA dataset to understand its characteristics.
    
    Args:
        split: Dataset split to analyze
        max_samples: Maximum number of samples to analyze (None for all)
    
    Returns:
        Dictionary with dataset statistics
    """
    logger.info(f"Analyzing {split} split...")
    
    try:
        dataset = load_dataset("vqa", "vqa_v2", data_dir=str(_DATA_DIR), split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Collect statistics
        question_lengths = []
        answer_counts = {}
        question_types = {}
        
        for sample in dataset:
            # Question length
            question = sample["question"]
            question_lengths.append(len(question.split()))
            
            # Question type (first word)
            first_word = question.split()[0].lower()
            question_types[first_word] = question_types.get(first_word, 0) + 1
            
            # Answer frequencies
            if isinstance(sample["answers"], dict):
                answers = sample["answers"]["answer"]
            else:
                answers = [ans["answer"] for ans in sample["answers"]]
            
            for answer in answers:
                answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        # Compile statistics
        stats = {
            "total_samples": len(dataset),
            "avg_question_length": np.mean(question_lengths),
            "question_length_std": np.std(question_lengths),
            "unique_answers": len(answer_counts),
            "most_common_answers": sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)[:20],
            "question_types": sorted(question_types.items(), key=lambda x: x[1], reverse=True)[:10],
            "question_length_stats": {
                "min": min(question_lengths),
                "max": max(question_lengths),
                "median": np.median(question_lengths)
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")
        raise


def print_dataset_stats(stats: Dict) -> None:
    """Print dataset statistics in a formatted way."""
    
    print("\n" + "="*50)
    print("VQA DATASET STATISTICS")
    print("="*50)
    
    print(f"\nTotal Samples: {stats['total_samples']:,}")
    print(f"Unique Answers: {stats['unique_answers']:,}")
    
    print(f"\nQuestion Length Statistics:")
    print(f"  Average: {stats['avg_question_length']:.2f} words")
    print(f"  Std Dev: {stats['question_length_std']:.2f} words")
    print(f"  Min: {stats['question_length_stats']['min']} words")
    print(f"  Max: {stats['question_length_stats']['max']} words")
    print(f"  Median: {stats['question_length_stats']['median']:.1f} words")
    
    print(f"\nTop Question Types:")
    for qtype, count in stats['question_types']:
        percentage = (count / stats['total_samples']) * 100
        print(f"  {qtype:10}: {count:6,} ({percentage:5.1f}%)")
    
    print(f"\nMost Common Answers:")
    for answer, count in stats['most_common_answers']:
        print(f"  {answer:15}: {count:6,}")
    
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="VQA Dataset Management")
    parser.add_argument("--download", action="store_true", help="Download VQA v2.0")
    parser.add_argument("--show-sample", action="store_true", help="Show a sample")
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset statistics")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"],
                       help="Dataset split to use")
    parser.add_argument("--index", type=int, default=0, help="Sample index to show")
    parser.add_argument("--max-samples", type=int, help="Max samples for analysis")
    parser.add_argument("--save-viz", type=str, help="Path to save sample visualization")
    
    args = parser.parse_args()

    if args.download:
        download_vqa()
    
    if args.show_sample:
        show_sample(args.split, args.index, args.save_viz)
    
    if args.analyze:
        stats = analyze_dataset(args.split, args.max_samples)
        print_dataset_stats(stats)


if __name__ == "__main__":
    main()
