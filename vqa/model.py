from transformers import CLIPModel, CLIPProcessor


class VQAModel:
    """Baseline VQA model using a pretrained CLIP model."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

    def forward(self, image, question):
        inputs = self.processor(text=[question], images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.logits_per_text
