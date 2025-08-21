import pytest
from vqa.train import collate_fn, VQADataset
from vqa.model import VQAModel

@pytest.fixture
def model():
    return VQAModel(num_answers=2)

def test_collate_fn(model):
    # Build a simple vocab for the model
    mock_dataset = [{"answers": [{"answer": "yes"}, {"answer": "no"}]}]
    model.build_answer_vocab(mock_dataset)
    
    batch = [
        {'image': None, 'question': 'test', 'answers': [{'answer': 'yes'}]},
        {'image': None, 'question': 'test2', 'answers': [{'answer': 'no'}]}
    ]
    collated = collate_fn(batch, model)
    assert 'images' in collated
    assert 'questions' in collated
    assert collated['targets'].shape == (2, model.classifier[-1].out_features)  # Dynamic num_answers

# Add more tests
