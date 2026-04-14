import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceEngine:
    def __init__(self):
        print("Loading Face Engine...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Face detector
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            keep_all=True,
            device=self.device
        )

        # Face embedding model
        self.resnet = InceptionResnetV1(
            pretrained='vggface2'
        ).eval().to(self.device)

        print(f"Face Engine loaded on {self.device}.")

    def extract_faces(self, image_path):

        image = Image.open(image_path).convert("RGB")

        faces = self.mtcnn(image)

        if faces is None:
            return []

        results = []

        for face_tensor in faces:

            # Ensure 3 channels (safety)
            if face_tensor.shape[0] == 1:
                face_tensor = face_tensor.repeat(3, 1, 1)

            face_tensor = face_tensor.to(self.device)

            embedding = self.resnet(
                face_tensor.unsqueeze(0)
            ).detach().cpu().numpy()[0]

            face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
            face_np = (face_np * 255).clip(0, 255).astype("uint8")
            face_pil = Image.fromarray(face_np)

            results.append({
                "embedding": embedding,
                "face_crop": face_pil
            })

        return results