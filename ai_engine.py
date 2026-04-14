import os
import threading


legacy_cache = os.environ.get("TRANSFORMERS_CACHE")
if legacy_cache and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = legacy_cache
os.environ.pop("TRANSFORMERS_CACHE", None)

from sentence_transformers import SentenceTransformer


class AIEngine:
    def __init__(self, preload=False):
        self.model_name = "intfloat/e5-base-v2"
        self.model = None
        self._lock = threading.Lock()
        self._loading_thread = None

        if preload:
            self._load_model()

    def _load_model(self):
        if self.model is not None:
            return

        with self._lock:
            if self.model is not None:
                return

            print("[TextEngine] Loading embedding model (E5-base)...")
            self.model = SentenceTransformer(self.model_name)
            print("[TextEngine] Model loaded successfully!")

    def warmup_async(self):
        if self.model is not None:
            return

        if self._loading_thread and self._loading_thread.is_alive():
            return

        self._loading_thread = threading.Thread(target=self._load_model, daemon=True)
        self._loading_thread.start()

    def text_to_vector(self, text: str, is_query=False):
        self._load_model()

        if is_query:
            text = "query: " + text
        else:
            text = "passage: " + text

        # normalize_embeddings=True makes cosine = dot product
        return self.model.encode(text, normalize_embeddings=True)