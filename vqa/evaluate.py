import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from datasets import load_dataset
from tqdm import tqdm

from .model import VQAModel
from .data import _DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def vqa_accuracy(predictions: List[str], ground_truths: List[List[str]]) -> float:
    """
    Calculate VQA accuracy as defined in the VQA paper.
    An answer is considered correct if at least 3 out of 10 annotators gave that answer.
    """
    correct = 0
    total = len(predictions)
    
    for pred, gt_list in zip(predictions, ground_truths):
        # Convert to lowercase for comparison
        pred = pred.lower().strip()
        gt_answers = [ans.lower().strip() for ans in gt_list]
        
        # Count occurrences of predicted answer in ground truth
        count = gt_answers.count(pred)
        
        # VQA accuracy: min(count/3, 1)
        accuracy = min(count / 3.0, 1.0)
        correct += accuracy
    
    return correct / total if total > 0 else 0.0


def analyze_by_question_type(
    predictions: List[str], 
    ground_truths: List[List[str]], 
    questions: List[str]
) -> Dict[str, float]:
    """Analyze accuracy by question type."""
    
    # Simple question type classification
    def classify_question(question: str) -> str:
        question = question.lower()
        if question.startswith('what'):
            return 'what'
        elif question.startswith('how many'):
            return 'count'
        elif question.startswith('is') or question.startswith('are') or question.startswith('does'):
            return 'yes/no'
        elif question.startswith('where'):
            return 'where'
        elif question.startswith('who'):
            return 'who'
        elif question.startswith('when'):
            return 'when'
        elif question.startswith('why'):
            return 'why'
        elif question.startswith('how'):
            return 'how'
        else:
            return 'other'
    
    # Group by question type
    type_predictions = defaultdict(list)
    type_ground_truths = defaultdict(list)
    
    for pred, gt, question in zip(predictions, ground_truths, questions):
        q_type = classify_question(question)
        type_predictions[q_type].append(pred)
        type_ground_truths[q_type].append(gt)
    
    # Calculate accuracy for each type
    type_accuracies = {}
    for q_type in type_predictions:
        acc = vqa_accuracy(type_predictions[q_type], type_ground_truths[q_type])
        type_accuracies[q_type] = acc
    
    return type_accuracies


def evaluate(
    model_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
    data_fraction: float = 0.1,
    batch_size: int = 32
) -> Dict[str, float]:
    """Evaluate the VQA model with proper metrics."""
    
    logger.info("Loading dataset...")
    val_split = f"validation[:{int(data_fraction * 100)}%]"
    dataset = load_dataset("vqa", "vqa_v2", data_dir=str(_DATA_DIR), split=val_split)
    
    # Initialize model
    logger.info("Initializing model...")
    model = VQAModel()
    
    # Load model weights if provided
    if model_path and Path(model_path).exists():
        logger.info(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # Load vocabulary
    if vocab_path and Path(vocab_path).exists():
        logger.info(f"Loading vocabulary from {vocab_path}")
        model.load_answer_vocab(vocab_path)
    else:
        logger.info("Building vocabulary from validation set...")
        model.build_answer_vocab(dataset)
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    logger.info(f"Evaluating on device: {device}")
    
    # Collect predictions and ground truths
    predictions = []
    ground_truths = []
    questions = []
    
    # Process in batches for efficiency
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch_end = min(i + batch_size, len(dataset))
        batch = dataset[i:batch_end]
        
        # Prepare batch data
        batch_images = [sample['image'] for sample in batch]
        batch_questions = [sample['question'] for sample in batch]
        batch_answers = [sample['answers'] for sample in batch]
        
        # Get predictions
        with torch.no_grad():
            batch_predictions = model.predict(batch_images, batch_questions, top_k=1)
        
        # Extract top prediction for each sample
        for j, sample_preds in enumerate(batch_predictions):
            if sample_preds:  # Check if predictions exist
                pred_answer = sample_preds[0][0]  # Top prediction
            else:
                pred_answer = "unknown"
            
            # Extract ground truth answers
            gt_answers = [ans['answer'] for ans in batch_answers[j]]
            
            predictions.append(pred_answer)
            ground_truths.append(gt_answers)
            questions.append(batch_questions[j])
    
    # Calculate overall accuracy
    overall_accuracy = vqa_accuracy(predictions, ground_truths)
    
    # Analyze by question type
    type_accuracies = analyze_by_question_type(predictions, ground_truths, questions)
    
    # Calculate answer frequency statistics
    answer_counts = defaultdict(int)
    for pred in predictions:
        answer_counts[pred] += 1
    
    # Most common predicted answers
    top_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Compile results
    results = {
        'overall_accuracy': overall_accuracy,
        'question_type_accuracies': type_accuracies,
        'total_samples': len(predictions),
        'top_predicted_answers': top_answers
    }
    
    return results


def print_results(results: Dict) -> None:
    """Print evaluation results in a formatted way."""
    
    print("\n" + "="*50)
    print("VQA EVALUATION RESULTS")
    print("="*50)
    
    print(f"\nOverall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Total Samples: {results['total_samples']}")
    
    print("\nAccuracy by Question Type:")
    print("-" * 30)
    for q_type, acc in sorted(results['question_type_accuracies'].items()):
        print(f"{q_type:12}: {acc:.4f}")
    
    print("\nTop Predicted Answers:")
    print("-" * 30)
    for answer, count in results['top_predicted_answers']:
        percentage = (count / results['total_samples']) * 100
        print(f"{answer:15}: {count:4d} ({percentage:5.1f}%)")
    
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA model")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--vocab-path", type=str, default="models/answer_vocab.json", 
                       help="Path to answer vocabulary")
    parser.add_argument("--data-fraction", type=float, default=0.1, 
                       help="Fraction of validation set to use")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate(
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        data_fraction=args.data_fraction,
        batch_size=args.batch_size
    )
    
    # Print results
    print_results(results)
    
    # Save results to file
    results_path = Path("evaluation_results.json")
    import json
    with open(results_path, 'w') as f:
        # Convert non-serializable items
        serializable_results = {
            'overall_accuracy': results['overall_accuracy'],
            'question_type_accuracies': results['question_type_accuracies'],
            'total_samples': results['total_samples'],
            'top_predicted_answers': results['top_predicted_answers']
        }
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
