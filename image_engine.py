import torch
import numpy as np
from PIL import Image
from transformers import SiglipProcessor, SiglipModel


def _unwrap_feature_tensor(output):
    """Normalize transformers output across versions to a tensor."""
    if hasattr(output, "cpu"):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Unsupported feature output type: {type(output).__name__}")


class ImageEngine:
    def __init__(self):
        print("Loading SigLIP model...")

        # Detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # ----------------------------
        # Load model to CPU first
        # ----------------------------
        if self.device == "cuda":
            self.model = SiglipModel.from_pretrained(
                "google/siglip-base-patch16-224",
                torch_dtype=torch.float16
            )
        else:
            self.model = SiglipModel.from_pretrained(
                "google/siglip-base-patch16-224"
            )

        print("Model loaded to CPU")

        # ----------------------------
        # Load processor BEFORE moving to CUDA
        # ----------------------------
        self.processor = SiglipProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )

        print("Processor loaded")

        # ----------------------------
        # Move model to device
        # ----------------------------
        if self.device == "cuda":
            self.model = self.model.to(self.device)
            print("Model moved to CUDA")

        self.model.eval()
        print("SigLIP model loaded successfully.")

    # ======================================================
    # IMAGE → VECTOR
    # ======================================================
    def image_to_vector(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.image_to_vector_from_pil(image)

    def image_to_vector_from_pil(self, image):
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        # Move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        feature_tensor = _unwrap_feature_tensor(outputs)
        vector = feature_tensor.cpu().numpy()[0]
        vector = vector / np.linalg.norm(vector)
        return vector

    # ======================================================
    # TEXT → VECTOR
    # ======================================================
    def text_to_vector(self, text):
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        feature_tensor = _unwrap_feature_tensor(outputs)
        vector = feature_tensor.cpu().numpy()[0]
        vector = vector / np.linalg.norm(vector)
        return vector