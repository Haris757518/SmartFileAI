"""
image_worker.py — Runs SigLIP in a completely isolated subprocess.

Why: SiglipProcessor (Rust tokenizers + SentencePiece) crashes silently
when loaded inside the same Windows process as PyQt6 + CUDA.
Running it in a separate process permanently solves this.

The worker receives requests via multiprocessing.Queue and returns
numpy vectors. If it crashes, the UI survives.
"""

import multiprocessing
import numpy as np
import traceback
import queue
import threading
import time
import os


def _unwrap_feature_tensor(output):
    """Normalize transformers output across versions to a tensor."""
    if hasattr(output, "cpu"):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Unsupported feature output type: {type(output).__name__}")


# ======================================================
# WORKER PROCESS (runs in isolation)
# ======================================================

def _worker_process(request_queue, response_queue):
    """Runs inside the isolated subprocess. Loads SigLIP once, then loops."""
    try:
        legacy_cache = os.environ.get("TRANSFORMERS_CACHE")
        if legacy_cache and "HF_HOME" not in os.environ:
            os.environ["HF_HOME"] = legacy_cache
        os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "models", "hf_cache"))
        os.environ.pop("TRANSFORMERS_CACHE", None)

        import torch
        from transformers import SiglipProcessor, SiglipModel
        from PIL import Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ImageWorker] Loading SigLIP on {device}...")

        if device == "cuda":
            model = SiglipModel.from_pretrained(
                "google/siglip-base-patch16-224",
                torch_dtype=torch.float16
            )
        else:
            model = SiglipModel.from_pretrained(
                "google/siglip-base-patch16-224"
            )

        processor = SiglipProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )

        if device == "cuda":
            model = model.to(device)

        model.eval()
        print("[ImageWorker] SigLIP ready.")

        response_queue.put({"status": "ready"})

        # ---- Main request loop ----
        while True:
            request = request_queue.get()

            if request is None:  # shutdown signal
                break

            req_type = request.get("type")

            try:
                if req_type == "image_to_vector":
                    image = Image.open(request["path"]).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.get_image_features(**inputs)

                    feature_tensor = _unwrap_feature_tensor(outputs)
                    vec = feature_tensor.cpu().float().numpy()[0]
                    vec = vec / np.linalg.norm(vec)
                    response_queue.put({"status": "ok", "vector": vec})

                elif req_type == "text_to_vector":
                    inputs = processor(
                        text=[request["text"]],
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.get_text_features(**inputs)

                    feature_tensor = _unwrap_feature_tensor(outputs)
                    vec = feature_tensor.cpu().float().numpy()[0]
                    vec = vec / np.linalg.norm(vec)
                    response_queue.put({"status": "ok", "vector": vec})

                else:
                    response_queue.put({"status": "error", "msg": f"Unknown type: {req_type}"})

            except Exception as e:
                response_queue.put({"status": "error", "msg": str(e)})

    except Exception as e:
        print(f"[ImageWorker] Fatal error: {e}")
        traceback.print_exc()
        response_queue.put({"status": "fatal", "msg": str(e)})


# ======================================================
# PROXY CLASS (used by AppCore instead of ImageEngine)
# ======================================================

class ImageWorkerProxy:
    """
    Drop-in replacement for ImageEngine.
    Communicates with SigLIP running in a separate process.
    """

    def __init__(self):
        self._process = None
        self._req_queue = None
        self._res_queue = None
        self._ready = False
        self._rpc_lock = threading.Lock()
        self._start_lock = threading.Lock()

    def start(self, wait_timeout=120):
        """Start the worker process. Call this once at app startup."""
        with self._start_lock:
            if self._process and self._process.is_alive():
                if wait_timeout and wait_timeout > 0:
                    self._wait_for_ready(wait_timeout)
                return

            ctx = multiprocessing.get_context("spawn")
            self._req_queue = ctx.Queue()
            self._res_queue = ctx.Queue()
            self._ready = False

            self._process = ctx.Process(
                target=_worker_process,
                args=(self._req_queue, self._res_queue),
                daemon=True
            )
            self._process.start()

            if wait_timeout and wait_timeout > 0:
                if not self._wait_for_ready(wait_timeout):
                    print("[ImageWorkerProxy] Worker still loading; continuing in background.")

    def _wait_for_ready(self, timeout):
        if self._ready:
            return True

        if not self._res_queue:
            return False

        if timeout is None:
            timeout = 0

        # For zero/negative timeout, do a non-blocking check so delayed
        # "ready" signals can still be consumed by polling code.
        if timeout <= 0:
            return self._consume_ready_signal(non_blocking=True)

        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = max(0.1, deadline - time.time())
            try:
                response = self._res_queue.get(timeout=remaining)
            except queue.Empty:
                break

            status = response.get("status")
            if status == "ready":
                self._ready = True
                print("[ImageWorkerProxy] Worker ready.")
                return True

            if status == "fatal":
                self._ready = False
                print(f"[ImageWorkerProxy] Worker fatal error: {response.get('msg')}")
                return False

        return self._ready

    def _consume_ready_signal(self, non_blocking=False):
        if self._ready or not self._res_queue:
            return self._ready

        try:
            response = self._res_queue.get_nowait() if non_blocking else self._res_queue.get(timeout=0.1)
        except queue.Empty:
            return self._ready

        status = response.get("status")
        if status == "ready":
            self._ready = True
            print("[ImageWorkerProxy] Worker ready.")
            return True

        if status == "fatal":
            self._ready = False
            print(f"[ImageWorkerProxy] Worker fatal error: {response.get('msg')}")
            return False

        return self._ready

    def ensure_ready(self, timeout=60):
        if self._ready:
            return True

        if not self._process or not self._process.is_alive():
            self.start(wait_timeout=timeout)
            return self._ready

        return self._wait_for_ready(timeout)

    def is_ready(self):
        with self._rpc_lock:
            if self._ready:
                return True
            return self._consume_ready_signal(non_blocking=True)

    def _request(self, payload, timeout=30):
        with self._rpc_lock:
            if not self.ensure_ready(timeout=timeout):
                print("[ImageWorkerProxy] Waiting for image model to finish loading...")
                if not self.ensure_ready(timeout=60):
                    raise RuntimeError("Image model failed to load.")

            self._req_queue.put(payload)

            try:
                response = self._res_queue.get(timeout=timeout)
            except queue.Empty:
                if self._process and not self._process.is_alive():
                    self._ready = False
                    raise RuntimeError("Image worker stopped unexpectedly.")
                raise RuntimeError("Image worker timed out while processing request.")

            if response.get("status") == "ok":
                return response["vector"]

            raise RuntimeError(f"ImageWorker error: {response.get('msg')}")

    def image_to_vector(self, image_path):
        return self._request({"type": "image_to_vector", "path": image_path}, timeout=30)

    def image_to_vector_from_pil(self, pil_image):
        """Save PIL to temp file, then process."""
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name
        try:
            pil_image.save(tmp_path, format="JPEG")
            return self.image_to_vector(tmp_path)
        finally:
            os.unlink(tmp_path)

    def text_to_vector(self, text):
        return self._request({"type": "text_to_vector", "text": text}, timeout=30)

    def shutdown(self):
        if self._req_queue:
            self._req_queue.put(None)
        if self._process:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
        self._ready = False