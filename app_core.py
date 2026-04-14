import numpy as np
import os
import time
import re
import threading
import json
import faiss

from ai_engine import AIEngine
from image_worker import ImageWorkerProxy
from database import DatabaseManager


class AppCore:
    def __init__(self, progress_callback=None):
        self._progress_callback = progress_callback

        self._report_progress(5, "Preparing local storage...")
        print("Initializing App Core...")

        os.makedirs("data", exist_ok=True)

        self.faiss_dir = os.path.join("data", "faiss")
        os.makedirs(self.faiss_dir, exist_ok=True)
        self.text_index_path = os.path.join(self.faiss_dir, "text.index")
        self.image_index_path = os.path.join(self.faiss_dir, "image.index")
        self.face_index_path = os.path.join(self.faiss_dir, "face.index")
        self.person_index_path = os.path.join(self.faiss_dir, "person.index")

        self.settings_path = os.path.join("data", "settings.json")
        self.settings = self._load_settings()
        self._background_preload_started = False
        self._background_preload_lock = threading.Lock()
        self._report_progress(15, "Initializing text engine...")

        # Load E5 lazily to keep startup responsive.
        self.text_engine = AIEngine(preload=False)
        self.text_engine.warmup_async()

        # ImageWorkerProxy replaces ImageEngine.
        # SigLIP runs in a separate process — no more Qt/CUDA crash.
        self.image_engine = ImageWorkerProxy()
        if self.is_background_ai_enabled():
            self.start_background_image_preload(force=True)

        self._face_query_engine = None

        self._report_progress(25, "Opening database...")
        self.db = DatabaseManager()

        self._report_progress(35, "Loading text index...")
        self.load_all_text_data(force_rebuild=False)
        self._report_progress(55, "Loading image index...")
        self.load_all_image_data(force_rebuild=False)
        self._report_progress(72, "Loading face index...")
        self.load_all_face_data(force_rebuild=False)
        self._report_progress(88, "Loading person index...")
        self.load_all_person_data(force_rebuild=False)

        self._report_progress(96, "Finalizing startup...")

        print("\nSearch Engine Ready.\n")
        self._report_progress(100, "Ready")
        self._progress_callback = None

    def _report_progress(self, percent, message):
        if not self._progress_callback:
            return

        try:
            self._progress_callback(int(percent), str(message))
        except Exception:
            pass

    def _load_settings(self):
        if not os.path.exists(self.settings_path):
            return {}

        try:
            with open(self.settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as e:
            print(f"[Settings] Failed to load settings: {e}")

        return {}

    def _save_settings(self):
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"[Settings] Failed to save settings: {e}")

    def should_prompt_background_ai(self):
        return "background_ai" not in self.settings

    def is_background_ai_enabled(self):
        return bool(self.settings.get("background_ai", False))

    def set_background_ai(self, enabled):
        self.settings["background_ai"] = bool(enabled)
        self._save_settings()

        if enabled:
            self.start_background_image_preload(force=True)

    def _get_setting_float(self, key, default, minimum=None, maximum=None):
        value = self.settings.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float(default)

        if minimum is not None:
            value = max(minimum, value)
        if maximum is not None:
            value = min(maximum, value)
        return value

    def get_image_text_min_score(self):
        return self._get_setting_float(
            "image_text_min_score",
            default=0.08,
            minimum=0.0,
            maximum=1.0,
        )

    def set_image_text_min_score(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.08

        value = max(0.0, min(1.0, value))
        self.settings["image_text_min_score"] = value
        self._save_settings()

    def start_background_image_preload(self, force=False):
        if not force and not self.is_background_ai_enabled():
            return

        with self._background_preload_lock:
            if self._background_preload_started:
                return
            self._background_preload_started = True

        def _preload():
            try:
                self.image_engine.ensure_ready(timeout=120)
            except Exception as e:
                print(f"[ImageWorker] Background preload failed: {e}")

        threading.Thread(target=_preload, daemon=True).start()

    def _load_or_build_index(self, index_path, embeddings, label, force_rebuild=False):
        expected_count = len(embeddings)
        expected_dim = embeddings.shape[1]

        if not force_rebuild and os.path.exists(index_path):
            try:
                index = faiss.read_index(index_path)
                if index.ntotal == expected_count and index.d == expected_dim:
                    print(f"{label} FAISS index loaded from disk.")
                    return index
                print(f"{label} FAISS index mismatch; rebuilding.")
            except Exception as e:
                print(f"{label} FAISS index load failed ({e}); rebuilding.")

        index = faiss.IndexFlatIP(expected_dim)
        index.add(embeddings)
        faiss.write_index(index, index_path)
        print(f"{label} FAISS index built and saved.")
        return index

    def refresh_all_indexes(self, force_rebuild=False):
        self.load_all_text_data(force_rebuild=force_rebuild)
        self.load_all_image_data(force_rebuild=force_rebuild)
        self.load_all_face_data(force_rebuild=force_rebuild)
        self.load_all_person_data(force_rebuild=force_rebuild)

    # ======================================================
    # LOAD TEXT DATA + BUILD FAISS
    # ======================================================
    def load_all_text_data(self, force_rebuild=False):
        print("Loading text embeddings...")

        data = self.db.fetch_all_text_chunks()

        self.file_paths = []
        self.file_names = []
        self.file_hashes = []
        self.last_modified = []
        self.chunk_texts = []
        self.text_embeddings = []

        for file_path, file_name, file_hash, last_modified, chunk_text, embedding in data:
            self.file_paths.append(file_path)
            self.file_names.append(file_name)
            self.file_hashes.append(file_hash)
            self.last_modified.append(last_modified)
            self.chunk_texts.append(chunk_text)
            self.text_embeddings.append(embedding)

        if self.text_embeddings:
            self.text_embeddings = np.array(self.text_embeddings).astype("float32")
            self.index = self._load_or_build_index(
                self.text_index_path,
                self.text_embeddings,
                "Text",
                force_rebuild=force_rebuild
            )
        else:
            self.text_embeddings = np.array([])
            self.index = None

        print(f"Loaded {len(self.text_embeddings)} text chunks.")

    # ======================================================
    # LOAD IMAGE DATA + BUILD FAISS
    # ======================================================
    def load_all_image_data(self, force_rebuild=False):
        print("Loading image embeddings...")

        data = self.db.fetch_all_images()

        self.image_paths = []
        self.image_names = []
        self.image_embeddings = []

        for file_path, file_name, file_hash, last_modified, embedding in data:
            self.image_paths.append(file_path)
            self.image_names.append(file_name)
            self.image_embeddings.append(embedding)

        if self.image_embeddings:
            self.image_embeddings = np.array(self.image_embeddings).astype("float32")
            self.image_index = self._load_or_build_index(
                self.image_index_path,
                self.image_embeddings,
                "Image",
                force_rebuild=force_rebuild
            )
        else:
            self.image_embeddings = np.array([])
            self.image_index = None

        print(f"Loaded {len(self.image_embeddings)} images.")

    # ======================================================
    # LOAD FACE DATA + BUILD FAISS
    # ======================================================
    def load_all_face_data(self, force_rebuild=False):
        print("Loading face embeddings...")

        data = self.db.fetch_all_faces()

        self.face_image_paths = []
        self.face_embeddings = []
        self.face_genders = []

        for file_path, embedding, gender in data:
            self.face_image_paths.append(file_path)
            self.face_embeddings.append(embedding)
            self.face_genders.append(gender)

        if self.face_embeddings:
            self.face_embeddings = np.array(self.face_embeddings).astype("float32")
            self.face_index = self._load_or_build_index(
                self.face_index_path,
                self.face_embeddings,
                "Face",
                force_rebuild=force_rebuild
            )
        else:
            self.face_embeddings = np.array([])
            self.face_index = None

        print(f"Loaded {len(self.face_embeddings)} faces.")

    # ======================================================
    # LOAD PERSON EMBEDDINGS + BUILD FAISS
    # ======================================================
    def load_all_person_data(self, force_rebuild=False):
        print("Loading person embeddings...")

        data = self.db.fetch_all_person_embeddings()

        self.person_image_paths = []
        self.person_embeddings = []

        for file_path, embedding in data:
            self.person_image_paths.append(file_path)
            self.person_embeddings.append(embedding)

        if self.person_embeddings:
            self.person_embeddings = np.array(self.person_embeddings).astype("float32")
            self.person_index = self._load_or_build_index(
                self.person_index_path,
                self.person_embeddings,
                "Person",
                force_rebuild=force_rebuild
            )
        else:
            self.person_embeddings = np.array([])
            self.person_index = None

        print(f"Loaded {len(self.person_embeddings)} person embeddings.")

    # ======================================================
    # BUILD TEXT FAISS
    # ======================================================
    def build_faiss_index(self):
        self.index = self._load_or_build_index(
            self.text_index_path,
            self.text_embeddings,
            "Text",
            force_rebuild=True
        )

    # ======================================================
    # EXTRACT VISUAL TERMS FROM QUERY
    # ======================================================
    def extract_visual_terms(self, query):
        stopwords = {
            "a", "an", "the", "wearing", "standing",
            "near", "with", "and", "in", "on",
            "at", "of", "to"
        }
        words = query.lower().split()
        meaningful = []
        current_phrase = []

        for word in words:
            if word in stopwords:
                if current_phrase:
                    meaningful.append(" ".join(current_phrase))
                    current_phrase = []
            else:
                current_phrase.append(word)

        if current_phrase:
            meaningful.append(" ".join(current_phrase))

        meaningful = [m for m in meaningful if len(m) > 2]
        return meaningful

    def _extract_image_search_intent(self, query):
        tokens = re.findall(r"[a-z0-9]+", query.lower())

        female_terms = {"woman", "women", "female", "girl", "girls", "lady", "ladies"}
        male_terms = {"man", "men", "male", "boy", "boys", "gentleman", "gentlemen"}
        people_terms = {
            "person", "people", "human", "face", "portrait", "selfie", "child", "children"
        }
        scene_terms = {
            "building", "buildings", "landslide", "earthquake", "collapse", "collapsed",
            "bridge", "road", "street", "city", "town", "village", "ocean", "sea",
            "wave", "river", "mountain", "forest", "tree", "nature", "house", "home",
            "tower", "temple", "car", "bus", "truck", "train", "plane", "airplane"
        }

        gender_filter = None
        if any(t in female_terms for t in tokens):
            gender_filter = "female"
        elif any(t in male_terms for t in tokens):
            gender_filter = "male"

        wants_people = None
        joined = " ".join(tokens)
        if "without people" in joined or "no people" in joined or "no person" in joined:
            wants_people = False
        elif any(t in female_terms or t in male_terms or t in people_terms for t in tokens):
            wants_people = True
        elif any(t in scene_terms for t in tokens):
            wants_people = False

        filler_terms = {
            "find", "show", "search", "images", "image", "photos", "photo", "pictures", "picture",
            "about", "of", "with", "and", "the", "a", "an", "in", "on", "at", "near", "for"
        }
        intent_terms = filler_terms.union(female_terms).union(male_terms).union(people_terms)
        topic_terms = [t for t in tokens if t not in intent_terms and len(t) >= 3]

        semantic_query = " ".join(topic_terms) if topic_terms else query
        return {
            "gender_filter": gender_filter,
            "wants_people": wants_people,
            "topic_terms": topic_terms,
            "semantic_query": semantic_query,
        }

    def _build_face_gender_map(self):
        face_gender_map = {}
        for path, gender in zip(self.face_image_paths, self.face_genders):
            key = os.path.normcase(os.path.normpath(path))
            if key not in face_gender_map:
                face_gender_map[key] = set()
            face_gender_map[key].add(gender)
        return face_gender_map

    # ======================================================
    # NORMALIZATION
    # ======================================================
    def normalize(self, text):
        text = text.lower()
        text = re.sub(r"[–—−]", "-", text)
        text = re.sub(r"[^a-z0-9]", "", text)
        return text

    def _extract_text_search_intent(self, query):
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        if not tokens:
            return set(), query, []

        type_keywords = {
            "pdf": {"pdf", "pdfs"},
            "word": {"doc", "docx", "word", "words"},
            "excel": {"xls", "xlsx", "excel", "spreadsheet", "spreadsheets", "csv"},
            "powerpoint": {"ppt", "pptx", "powerpoint", "slide", "slides"},
            "text": {"txt", "text", "plaintext", "md", "markdown"},
            "json": {"json"},
            "xml": {"xml"},
            "code": {"code", "python", "py", "javascript", "js", "java", "cpp", "c", "cs", "ts", "go", "rs", "php"},
        }
        type_extensions = {
            "pdf": {".pdf"},
            "word": {".doc", ".docx"},
            "excel": {".xls", ".xlsx", ".csv"},
            "powerpoint": {".ppt", ".pptx"},
            "text": {".txt", ".md", ".rtf", ".log"},
            "json": {".json"},
            "xml": {".xml"},
            "code": {
                ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp",
                ".h", ".hpp", ".cs", ".go", ".rs", ".php", ".rb", ".swift"
            },
        }

        intent_tokens = set()
        allowed_extensions = set()
        for group, keywords in type_keywords.items():
            matched = keywords.intersection(tokens)
            if matched:
                intent_tokens.update(matched)
                allowed_extensions.update(type_extensions[group])

        stopwords = {
            "find", "show", "search", "give", "me", "about", "on", "for", "with",
            "related", "only", "just", "file", "files", "document", "documents",
            "doc", "docs", "the", "a", "an", "to", "from", "of", "and", "in"
        }
        topic_terms = [
            t for t in tokens
            if t not in intent_tokens and t not in stopwords and len(t) >= 3
        ]

        semantic_query = " ".join(topic_terms) if topic_terms else query
        return allowed_extensions, semantic_query, topic_terms

    def _search_files_by_extension(self, allowed_extensions, top_k=5):
        if not allowed_extensions:
            return []

        file_results = {}
        current_time = time.time()

        for idx, file_path in enumerate(self.file_paths):
            file_name = self.file_names[idx]
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in allowed_extensions:
                continue

            age_days = (current_time - self.last_modified[idx]) / 86400
            recency_score = max(0.0, 1.0 - (age_days / (5 * 365)))
            final_score = 0.6 + 0.4 * recency_score

            if file_path not in file_results or final_score > file_results[file_path]["final_score"]:
                file_results[file_path] = {
                    "file_path": file_path,
                    "file_name": file_name,
                    "final_score": float(final_score),
                    "snippet": self.chunk_texts[idx][:300],
                }

        if not file_results:
            return []

        sorted_results = sorted(file_results.values(), key=lambda x: x["final_score"], reverse=True)
        return sorted_results[:top_k]

    def _search_files_by_topic_and_extension(self, topic_terms, allowed_extensions, top_k=5):
        normalized_terms = [self.normalize(t) for t in topic_terms if self.normalize(t)]
        if not normalized_terms or not allowed_extensions:
            return []

        file_results = {}
        current_time = time.time()
        term_count = len(normalized_terms)

        for idx, file_path in enumerate(self.file_paths):
            file_name = self.file_names[idx]
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext not in allowed_extensions:
                continue

            file_name_norm = self.normalize(file_name)
            chunk_norm = self.normalize(self.chunk_texts[idx])

            name_hits = sum(1 for term in normalized_terms if term in file_name_norm)
            chunk_hits = sum(1 for term in normalized_terms if term in chunk_norm)
            total_hits = max(name_hits, chunk_hits)
            if total_hits == 0:
                continue

            coverage = total_hits / term_count
            file_last_modified = self.last_modified[idx]
            age_days = (current_time - file_last_modified) / 86400
            recency_boost = max(0, 0.05 - (age_days / (5 * 365)) * 0.05)

            # Prioritize explicit topic mentions before recency.
            final_score = (
                2.2 * coverage +
                0.3 * (name_hits / term_count) +
                0.2 * (chunk_hits / term_count) +
                recency_boost
            )

            if file_path not in file_results or final_score > file_results[file_path]["final_score"]:
                file_results[file_path] = {
                    "file_path": file_path,
                    "file_name": file_name,
                    "final_score": float(final_score),
                    "snippet": self.chunk_texts[idx][:300],
                }

        if not file_results:
            return []

        sorted_results = sorted(file_results.values(), key=lambda x: x["final_score"], reverse=True)
        return sorted_results[:top_k]

    # ======================================================
    # TEXT SEARCH
    # ======================================================
    def search_text(self, query, top_k=5):
        if len(self.text_embeddings) == 0 or self.index is None:
            return []

        allowed_extensions, semantic_query, topic_terms = self._extract_text_search_intent(query)
        strict_topic_type_mode = bool(allowed_extensions and topic_terms)

        if strict_topic_type_mode:
            return self._search_files_by_topic_and_extension(
                topic_terms,
                allowed_extensions,
                top_k=top_k,
            )

        query_norm = self.normalize(semantic_query)
        if len(query_norm) < 3 and not allowed_extensions:
            return []

        if len(query_norm) < 3 and allowed_extensions and not topic_terms:
            return self._search_files_by_extension(allowed_extensions, top_k=top_k)

        query_vector = self.text_engine.text_to_vector(semantic_query, is_query=True)
        query_vector = np.array([query_vector]).astype("float32")

        candidate_k = min(max(top_k * 30, 120), len(self.text_embeddings))
        D, I = self.index.search(query_vector, candidate_k)
        if len(D[0]) == 0:
            return []

        top_semantic = None
        for rank, idx in enumerate(I[0]):
            semantic_score = float(D[0][rank])
            if not np.isfinite(semantic_score):
                continue

            if allowed_extensions:
                ext = os.path.splitext(self.file_names[idx])[1].lower()
                if ext not in allowed_extensions:
                    continue

            top_semantic = semantic_score
            break

        if top_semantic is None:
            return self._search_files_by_extension(allowed_extensions, top_k=top_k)

        file_results = {}
        current_time = time.time()

        is_identifier = False
        if any(c.isdigit() for c in query_norm):
            is_identifier = True
        elif len(query_norm) <= 6 and not query_norm.isalpha():
            is_identifier = True

        normalized_terms = [self.normalize(t) for t in topic_terms if self.normalize(t)]
        term_count = max(1, len(normalized_terms))
        strict_single_term_match = (
            not allowed_extensions and
            len(normalized_terms) == 1 and
            len(normalized_terms[0]) >= 4 and
            normalized_terms[0].isalpha()
        )

        for rank, idx in enumerate(I[0]):
            semantic_score = float(D[0][rank])
            if not np.isfinite(semantic_score):
                continue

            if semantic_score < top_semantic - 0.25:
                continue

            chunk_text = self.chunk_texts[idx]
            file_path = self.file_paths[idx]
            file_name = self.file_names[idx]
            file_ext = os.path.splitext(file_name)[1].lower()

            if allowed_extensions and file_ext not in allowed_extensions:
                continue

            file_name_norm = self.normalize(file_name)
            chunk_norm = self.normalize(chunk_text)

            if normalized_terms:
                name_hits = sum(1 for t in normalized_terms if t in file_name_norm)
                chunk_hits = sum(1 for t in normalized_terms if t in chunk_norm)
                filename_match = name_hits / term_count
                lexical_match = chunk_hits / term_count
                if len(query_norm) >= 4 and query_norm in chunk_norm:
                    lexical_match = max(lexical_match, 1.0)
            else:
                filename_match = 1.0 if query_norm in file_name_norm and query_norm else 0.0
                lexical_match = 1.0 if query_norm in chunk_norm and query_norm else 0.0

            # For single word queries like names, require an actual lexical hit
            # so misspellings or unseen terms do not return unrelated documents.
            if strict_single_term_match and filename_match == 0 and lexical_match == 0:
                continue

            file_last_modified = self.last_modified[idx]
            age_days = (current_time - file_last_modified) / 86400
            recency_boost = max(0, 0.05 - (age_days / (5 * 365)) * 0.05)
            type_bonus = 0.15 if allowed_extensions else 0.0

            if is_identifier:
                if filename_match == 0 and lexical_match == 0:
                    continue
                final_score = (
                    2.5 * filename_match +
                    2.0 * lexical_match +
                    0.5 * semantic_score +
                    type_bonus +
                    recency_boost
                )
            else:
                if filename_match == 0 and lexical_match == 0 and semantic_score < top_semantic - 0.08:
                    continue
                final_score = (
                    1.1 * filename_match +
                    1.4 * lexical_match +
                    1.35 * semantic_score +
                    type_bonus +
                    recency_boost
                )

            if file_path not in file_results or final_score > file_results[file_path]["final_score"]:
                file_results[file_path] = {
                    "file_path": file_path,
                    "file_name": file_name,
                    "final_score": float(final_score),
                    "snippet": chunk_text[:300]
                }

        if not file_results:
            return self._search_files_by_extension(allowed_extensions, top_k=top_k)

        sorted_results = sorted(file_results.values(), key=lambda x: x["final_score"], reverse=True)
        return sorted_results[:top_k]

    # ======================================================
    # IMAGE SEARCH BY TEXT
    # ======================================================
    def search_images_by_text(self, query, top_k=10):
        if self.image_index is None:
            return []

        intent = self._extract_image_search_intent(query)
        gender_filter = intent["gender_filter"]
        wants_people = intent["wants_people"]
        topic_terms = intent["topic_terms"]
        semantic_query = intent["semantic_query"]

        terms = self.extract_visual_terms(semantic_query)
        if not terms:
            terms = [semantic_query if semantic_query.strip() else query]

        include_person_index = (
            self.person_index is not None and
            len(self.person_embeddings) > 0 and
            wants_people is not False and
            gender_filter is None
        )

        results_map = {}

        for term in terms:
            query_vector = self.image_engine.text_to_vector(term)
            query_vector = np.array([query_vector]).astype("float32")

            image_k = min(max(top_k * 10, 100), len(self.image_embeddings))
            D_img, I_img = self.image_index.search(query_vector, image_k)
            for rank, idx in enumerate(I_img[0]):
                score = float(D_img[0][rank])
                if not np.isfinite(score):
                    continue

                path = self.image_paths[idx]
                if path not in results_map:
                    results_map[path] = {"image_scores": [], "person_scores": []}
                results_map[path]["image_scores"].append(score)

            if include_person_index:
                person_k = min(max(top_k * 12, 120), len(self.person_embeddings))
                D_person, I_person = self.person_index.search(query_vector, person_k)
                for rank, idx in enumerate(I_person[0]):
                    score = float(D_person[0][rank])
                    if not np.isfinite(score):
                        continue

                    path = self.person_image_paths[idx]
                    if path not in results_map:
                        results_map[path] = {"image_scores": [], "person_scores": []}
                    results_map[path]["person_scores"].append(score)

        if not results_map:
            return []

        normalized_terms = [self.normalize(t) for t in topic_terms if self.normalize(t)]
        term_count = max(1, len(normalized_terms))

        final_results = []
        for path, score_data in results_map.items():
            image_scores = score_data["image_scores"]
            person_scores = score_data["person_scores"]

            if not image_scores and not person_scores:
                continue

            image_best = max(image_scores) if image_scores else 0.0
            image_mean = (sum(image_scores) / len(image_scores)) if image_scores else image_best
            person_best = max(person_scores) if person_scores else 0.0
            person_mean = (sum(person_scores) / len(person_scores)) if person_scores else person_best

            if wants_people is True:
                base_score = 0.45 * image_best + 0.55 * max(person_best, person_mean)
                if person_scores:
                    base_score += 0.03
            elif wants_people is False:
                if not image_scores:
                    continue
                base_score = 0.9 * max(image_best, image_mean) + 0.1 * person_best
                if person_scores and person_best > image_best:
                    base_score -= 0.03
            else:
                base_score = 0.72 * max(image_best, image_mean) + 0.28 * max(person_best, person_mean)

            file_name = os.path.basename(path)
            file_name_norm = self.normalize(file_name)
            lexical_bonus = 0.0
            if normalized_terms:
                name_hits = sum(1 for t in normalized_terms if t and t in file_name_norm)
                lexical_bonus = 0.12 * (name_hits / term_count)

            final_score = base_score + lexical_bonus
            final_results.append({
                "file_path": path,
                "file_name": file_name,
                "final_score": float(final_score)
            })

        final_results = sorted(final_results, key=lambda x: x["final_score"], reverse=True)

        if gender_filter and final_results:
            face_gender_map = self._build_face_gender_map()
            opposite = "male" if gender_filter == "female" else "female"

            strict_filtered = []
            for result in final_results:
                key = os.path.normcase(os.path.normpath(result["file_path"]))
                genders = face_gender_map.get(key, set())
                if genders and gender_filter in genders and opposite not in genders:
                    strict_filtered.append(result)

            if strict_filtered:
                final_results = strict_filtered
            else:
                relaxed_filtered = []
                for result in final_results:
                    key = os.path.normcase(os.path.normpath(result["file_path"]))
                    genders = face_gender_map.get(key, set())
                    if genders and gender_filter in genders:
                        relaxed_filtered.append(result)
                final_results = relaxed_filtered

        if not final_results:
            return []

        top_score = final_results[0]["final_score"]
        if gender_filter and top_score < 0.06:
            return []

        if wants_people is True and top_score < 0.02:
            return []

        min_score = self.get_image_text_min_score()
        if wants_people is True:
            margin = 0.10
        elif wants_people is False:
            margin = 0.08
        else:
            margin = 0.09

        adaptive_floor = min(min_score, max(0.01, top_score * 0.85))
        dynamic_cutoff = max(adaptive_floor, top_score - margin)
        filtered_results = [
            r for r in final_results
            if r["final_score"] >= dynamic_cutoff
        ]

        if not filtered_results:
            fallback_gate = 0.02 if wants_people is True else max(0.04, min_score * 0.7)
            if top_score < fallback_gate:
                return []
            keep_n = min(len(final_results), max(2, min(top_k, 4)))
            return final_results[:keep_n]

        final_results = filtered_results

        return final_results[:top_k]

    # ======================================================
    # IMAGE SIMILARITY SEARCH
    # ======================================================
    def search_similar_images(self, image_path, top_k=10):
        if self.image_index is None:
            return []

        query_vector = self.image_engine.image_to_vector(image_path)
        query_vector = np.array([query_vector]).astype("float32")

        D, I = self.image_index.search(query_vector, min(top_k + 1, len(self.image_embeddings)))

        results = []
        for rank, idx in enumerate(I[0]):
            path = self.image_paths[idx]
            if os.path.normpath(path) == os.path.normpath(image_path):
                continue
            results.append({
                "file_path": path,
                "file_name": os.path.basename(path),
                "final_score": float(D[0][rank])
            })

        return results[:top_k]

    # ======================================================
    # FACE SEARCH — BUG FIXED: extract embedding from dict
    # ======================================================
    def search_by_face(self, image_path, top_k=10):
        if self.face_index is None or len(self.face_embeddings) == 0:
            return []

        if self._face_query_engine is None:
            from face_engine import FaceEngine
            self._face_query_engine = FaceEngine()

        query_faces = self._face_query_engine.extract_faces(image_path)

        if not query_faces:
            print("No face detected in query image.")
            return []

        # BUG FIX: extract_faces returns dicts with "embedding" key
        query_vector = np.array([query_faces[0]["embedding"]]).astype("float32")

        D, I = self.face_index.search(query_vector, min(20, len(self.face_embeddings)))

        if len(D[0]) == 0:
            return []

        results = []
        for rank, idx in enumerate(I[0]):
            score = float(D[0][rank])
            if score < 0.65:
                continue
            results.append({
                "file_path": self.face_image_paths[idx],
                "file_name": os.path.basename(self.face_image_paths[idx]),
                "final_score": score
            })

        return results[:top_k]

    # ======================================================
    # LIVE UPDATE FROM WATCHER
    # ======================================================
    def reindex_single_file(self, file_path):
        from indexer import FileIndexer
        indexer = FileIndexer()
        indexer.process_file(file_path)
        self.load_all_text_data(force_rebuild=True)
        print("[FAISS] Index rebuilt after update.")

    def remove_file_from_index(self, file_path):
        self.db.delete_file_chunks(file_path)
        self.db.delete_image(file_path)
        self.db.delete_face_embeddings(file_path)
        self.db.delete_person_embeddings(file_path)
        self.refresh_all_indexes(force_rebuild=True)
        print("[FAISS] Index rebuilt after deletion.")

    # ======================================================
    # OPEN FILE
    # ======================================================
    def open_file(self, file_path):
        if os.path.exists(file_path):
            os.startfile(file_path)
        else:
            print("File does not exist.")

    def find_duplicate_images(self, folder_path, mode="hybrid", phash_threshold=8):
        from duplicate_detector import scan_folder_duplicates
        return scan_folder_duplicates(
            folder_path,
            mode=mode,
            phash_threshold=phash_threshold,
        )

    def shutdown(self):
        try:
            self.image_engine.shutdown()
        except Exception:
            pass

    # ======================================================
    # STATS FOR DASHBOARD
    # ======================================================
    def get_stats(self):
        return {
            "text_chunks": len(self.text_embeddings) if len(self.text_embeddings) > 0 else 0,
            "images": len(self.image_embeddings) if len(self.image_embeddings) > 0 else 0,
            "faces": len(self.face_embeddings) if len(self.face_embeddings) > 0 else 0,
            "persons": len(self.person_embeddings) if len(self.person_embeddings) > 0 else 0,
        }