import torch
import torchvision
import numpy as np
from PIL import Image


class PersonEngine:
    def __init__(self):
        print("Loading Person Detection Engine...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        ).to(self.device)

        self.model.eval()

        print(f"Person Detection loaded on {self.device}.")

    def detect_person_crops(self, image_path, score_threshold=0.7):
        """
        Returns list of cropped PIL images containing persons.
        """
        image = Image.open(image_path).convert("RGB")

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

        input_tensor = transform(image).to(self.device)

        with torch.no_grad():
            outputs = self.model([input_tensor])[0]

        boxes = outputs["boxes"]
        labels = outputs["labels"]
        scores = outputs["scores"]

        person_crops = []

        for box, label, score in zip(boxes, labels, scores):

            # COCO class 1 = person
            if label.item() == 1 and score.item() >= score_threshold:
                x1, y1, x2, y2 = box.int().tolist()

                crop = image.crop((x1, y1, x2, y2))
                person_crops.append(crop)

        return person_crops