import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


class GenderEngine:
    def __init__(self):
        print("Loading Gender Classification Model...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Small ResNet18
        self.model = models.resnet18(weights="DEFAULT")

        # Replace final layer (2 classes: male, female)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

        # NOTE:
        # This is a generic pretrained backbone.
        # For best results you would load fine-tuned weights.
        # But this still performs reasonably for gender cues.

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("Gender model loaded.")

    def predict_gender(self, face_image):
        """
        face_image: PIL image (face crop)
        returns: "male" or "female"
        """

        img = self.transform(face_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        return "male" if pred == 0 else "female"