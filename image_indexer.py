import os
import hashlib
from PIL import Image
from image_engine import ImageEngine
from database import DatabaseManager
from face_engine import FaceEngine
from person_engine import PersonEngine
from gender_engine import GenderEngine
class ImageIndexer:
    # All formats PIL can handle + common image formats
    SUPPORTED_FORMATS = (
        ".jpg", ".jpeg", ".png", ".webp",
        ".bmp", ".gif", ".tiff", ".tif",
        ".heic", ".heif", ".ico", ".jfif",
        ".ppm", ".pgm", ".pbm", ".pnm",
    )

    def __init__(self):
        print("Initializing Image Indexer...")
        self.engine = ImageEngine()
        self.face_engine = FaceEngine()
        self.person_engine = PersonEngine()
        self.gender_engine = GenderEngine()
        self.db = DatabaseManager()

    def index_folder(self, folder_path):
        print(f"Scanning for images in: {folder_path}")

        # Load all indexed hashes ONCE up-front
        print("Loading existing index...")
        existing_records = self.db.fetch_all_images()
        indexed_hashes = {record[2] for record in existing_records}
        indexed_paths = {record[0] for record in existing_records}

        new_count = 0
        skip_count = 0
        error_count = 0

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(self.SUPPORTED_FORMATS):
                    full_path = os.path.join(root, file)
                    result = self.process_image(full_path, indexed_hashes, indexed_paths)
                    if result == "indexed":
                        new_count += 1
                        indexed_hashes.add(self.compute_hash(full_path))
                        indexed_paths.add(full_path)
                    elif result == "skipped":
                        skip_count += 1
                    elif result == "error":
                        error_count += 1

        # Remove deleted files from DB
        disk_files = set()
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(self.SUPPORTED_FORMATS):
                    disk_files.add(os.path.join(root, file))

        deleted = indexed_paths - disk_files
        for deleted_path in deleted:
            print(f"Removing deleted image: {deleted_path}")
            self.db.delete_image(deleted_path)
            self.db.delete_face_embeddings(deleted_path)
            self.db.delete_person_embeddings(deleted_path)

        print(f"\nImage indexing completed.")
        print(f"  New:     {new_count}")
        print(f"  Skipped: {skip_count}")
        print(f"  Errors:  {error_count}")
        print(f"  Deleted: {len(deleted)}")

    def process_image(self, file_path, indexed_hashes=None, indexed_paths=None):
        try:
            file_name = os.path.basename(file_path)
            last_modified = os.path.getmtime(file_path)
            file_hash = self.compute_hash(file_path)

            # Fast O(1) lookup
            if indexed_hashes is not None and file_hash in indexed_hashes:
                print(f"Skipping (already indexed): {file_name}")
                return "skipped"

            if indexed_hashes is None:
                existing = self.db.fetch_all_images()
                if any(record[2] == file_hash for record in existing):
                    print(f"Skipping (already indexed): {file_name}")
                    return "skipped"

            # Convert to RGB — handles webp, heic, palette, RGBA, etc.
            image = Image.open(file_path)
            image = image.convert("RGB")

            image_embedding = self.engine.image_to_vector_from_pil(image)
            # -----------------------------
            # PERSON DETECTION + EMBEDDING
            # -----------------------------
            person_crops = self.person_engine.detect_person_crops(file_path)

            for crop in person_crops:
                crop_embedding = self.engine.image_to_vector_from_pil(crop)
                self.db.insert_person_embedding(file_path, crop_embedding)
            # -----------------------------
            # FACE DETECTION + EMBEDDING
            # -----------------------------
            faces = self.face_engine.extract_faces(file_path)

            for face in faces:
                face_embedding = face["embedding"]
                face_crop = face["face_crop"]
                gender = self.gender_engine.predict_gender(face_crop)
                self.db.insert_face_embedding(file_path, face_embedding, gender)
            self.db.insert_image(
                file_path,
                file_name,
                file_hash,
                last_modified,
                image_embedding
            )

            print(f"Indexed: {file_name}")
            return "indexed"

        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
            return "error"

    def compute_hash(self, file_path):
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(4096):
                hasher.update(chunk)
        return hasher.hexdigest()