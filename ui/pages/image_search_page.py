import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit,
    QScrollArea, QFrame, QGridLayout,
    QSizePolicy, QFileDialog, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QPixmap


# ======================================================
# WORKER
# ======================================================
class ImageSearchWorker(QThread):
    results_ready = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, app_core, mode, query=None, image_path=None):
        super().__init__()
        self.app_core = app_core
        self.mode = mode          # "text" or "similar"
        self.query = query
        self.image_path = image_path

    def run(self):
        for attempt in range(2):
            try:
                if self.mode == "text":
                    results = self.app_core.search_images_by_text(self.query, top_k=20)
                else:
                    results = self.app_core.search_similar_images(self.image_path, top_k=20)
                self.results_ready.emit(results)
                return
            except Exception as e:
                msg = str(e)
                if "still loading" in msg.lower() and attempt == 0:
                    self.msleep(1500)
                    continue
                self.error.emit(msg)
                return


# ======================================================
# IMAGE THUMBNAIL CARD
# ======================================================
class ImageCard(QWidget):
    def __init__(self, result, app_core, thumb_size=180):
        super().__init__()
        self.app_core = app_core
        self.file_path = result["file_path"]
        self.setObjectName("ImageCard")
        self.setFixedSize(thumb_size + 16, thumb_size + 52)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        self.setLayout(layout)

        # Thumbnail
        img_label = QLabel()
        img_label.setFixedSize(thumb_size, thumb_size)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label.setStyleSheet("border-radius: 6px; background-color: #0B1118;")

        if os.path.exists(self.file_path):
            pix = QPixmap(self.file_path)
            if not pix.isNull():
                pix = pix.scaled(
                    thumb_size, thumb_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                img_label.setPixmap(pix)
            else:
                img_label.setText("🖼")
                img_label.setStyleSheet("font-size: 32px; color: #3C556D; background-color: #0B1118;")
        else:
            img_label.setText("✗")
            img_label.setStyleSheet("font-size: 24px; color: #3C556D; background-color: #0B1118;")

        layout.addWidget(img_label)

        # Score + name row
        bottom = QHBoxLayout()
        bottom.setSpacing(4)

        score_val = result.get("final_score", 0)
        score = QLabel(f"{score_val:.2f}")
        score.setObjectName("ScoreBadge")
        bottom.addWidget(score)

        name = QLabel(result.get("file_name", "")[:22])
        name.setObjectName("ImageLabel")
        name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bottom.addWidget(name, 1)

        layout.addLayout(bottom)

    def mouseDoubleClickEvent(self, event):
        self.app_core.open_file(self.file_path)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Single click opens too (user-friendly for images)
            self.app_core.open_file(self.file_path)


# ======================================================
# PAGE
# ======================================================
class ImageSearchPage(QWidget):
    def __init__(self, app_core):
        super().__init__()
        self.app_core = app_core
        self._worker = None
        self._selected_image = None
        self._pending_search = None
        self._startup_triggered = False

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 36, 40, 20)
        layout.setSpacing(0)
        self.setLayout(layout)

        # Header
        title = QLabel("Image Search")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel("Search by text description or upload an image to find similar ones")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(24)

        # Search controls
        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)

        self.search_input = QLineEdit()
        self.search_input.setObjectName("SearchInput")
        self.search_input.setPlaceholderText("Describe what you're looking for…")
        self.search_input.setMinimumHeight(44)
        self.search_input.returnPressed.connect(self._run_text_search)
        controls_row.addWidget(self.search_input, 1)

        self.search_btn = QPushButton("Search")
        self.search_btn.setObjectName("PrimaryButton")
        self.search_btn.setMinimumHeight(44)
        self.search_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.search_btn.clicked.connect(self._run_text_search)
        controls_row.addWidget(self.search_btn)

        upload_btn = QPushButton("Upload Image")
        upload_btn.setObjectName("SecondaryButton")
        upload_btn.setMinimumHeight(44)
        upload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        upload_btn.clicked.connect(self._upload_image)
        controls_row.addWidget(upload_btn)

        layout.addLayout(controls_row)
        layout.addSpacing(12)

        # Selected image preview row
        self.preview_row = QHBoxLayout()
        self.preview_row.setSpacing(10)

        self.preview_label = QLabel()
        self.preview_label.hide()
        self.preview_row.addWidget(self.preview_label)

        self.similar_btn = QPushButton("Find Similar Images")
        self.similar_btn.setObjectName("PrimaryButton")
        self.similar_btn.setMinimumHeight(38)
        self.similar_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.similar_btn.clicked.connect(self._run_similar_search)
        self.similar_btn.hide()
        self.preview_row.addWidget(self.similar_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("SecondaryButton")
        self.clear_btn.setMinimumHeight(38)
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.clicked.connect(self._clear_image)
        self.clear_btn.hide()
        self.preview_row.addWidget(self.clear_btn)

        self.preview_row.addStretch()
        layout.addLayout(self.preview_row)

        # Status
        self.status_label = QLabel("")
        self.status_label.setObjectName("LoadingLabel")
        layout.addWidget(self.status_label)

        layout.addSpacing(8)

        # Results scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.results_container = QWidget()
        self.results_layout = QGridLayout()
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(12)
        self.results_container.setLayout(self.results_layout)
        self.scroll.setWidget(self.results_container)
        layout.addWidget(self.scroll, 1)

        self._show_empty_state()

        self._model_state_timer = QTimer(self)
        self._model_state_timer.setInterval(700)
        self._model_state_timer.timeout.connect(self._refresh_model_state)
        self._model_state_timer.start()
        self._refresh_model_state()

    def _trigger_model_startup(self):
        if self._startup_triggered:
            return

        self._startup_triggered = True
        try:
            # Start worker asynchronously; readiness is tracked by timer polling.
            self.app_core.image_engine.start(wait_timeout=0)
        except Exception as e:
            self.status_label.setText(f"Error: {e}")

    def _refresh_model_state(self):
        if self._worker and self._worker.isRunning():
            return

        model_ready = self.app_core.image_engine.is_ready()
        if model_ready:
            if self.search_btn.text() == "Loading...":
                self.search_btn.setText("Search")
            self.search_btn.setEnabled(True)
            self.similar_btn.setEnabled(self._selected_image is not None)

            if self.status_label.text().startswith("Loading AI model"):
                self.status_label.setText("")

            if self._pending_search is not None:
                mode, query, image_path = self._pending_search
                self._pending_search = None
                self._start_search(mode, query=query, image_path=image_path)
                return

            self._model_state_timer.stop()
        else:
            self._trigger_model_startup()
            self.search_btn.setEnabled(False)
            self.search_btn.setText("Loading...")
            self.similar_btn.setEnabled(False)

            if not self.status_label.text() or self.status_label.text().startswith("Error:"):
                self.status_label.setText("Preparing AI model... (first time only)")

    def _show_empty_state(self):
        self._clear_results()
        empty = QLabel("Search by description or upload an image")
        empty.setObjectName("EmptyStateLabel")
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(empty, 0, 0, 1, 5)

    def _clear_results(self):
        while self.results_layout.count() > 0:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _upload_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.jpg *.jpeg *.png *.webp *.bmp *.gif *.tiff)"
        )
        if path:
            self._selected_image = path
            pix = QPixmap(path).scaled(
                60, 60,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_label.setPixmap(pix)
            self.preview_label.show()
            self.similar_btn.show()
            self.clear_btn.show()
            self.status_label.setText(f"Image selected: {os.path.basename(path)}")

    def _clear_image(self):
        self._selected_image = None
        self.preview_label.clear()
        self.preview_label.hide()
        self.similar_btn.hide()
        self.clear_btn.hide()
        self.status_label.setText("")

    def _run_text_search(self):
        query = self.search_input.text().strip()
        if not query:
            return

        if not self.app_core.image_engine.is_ready():
            self._trigger_model_startup()
            self._pending_search = ("text", query, None)
            self.status_label.setText("Preparing AI model... your search will run automatically.")
            if not self._model_state_timer.isActive():
                self._model_state_timer.start()
            return

        self._start_search("text", query=query)

    def _run_similar_search(self):
        if not self._selected_image:
            return

        if not self.app_core.image_engine.is_ready():
            self._trigger_model_startup()
            self._pending_search = ("similar", None, self._selected_image)
            self.status_label.setText("Preparing AI model... similar search will run automatically.")
            if not self._model_state_timer.isActive():
                self._model_state_timer.start()
            return

        self._start_search("similar", image_path=self._selected_image)

    def _start_search(self, mode, query=None, image_path=None):
        if self._worker and self._worker.isRunning():
            return

        self.search_btn.setEnabled(False)
        self.similar_btn.setEnabled(False)
        self.search_btn.setText("Searching…")
        self.status_label.setText("Searching…")
        self._clear_results()

        self._worker = ImageSearchWorker(self.app_core, mode, query, image_path)
        self._worker.results_ready.connect(self._show_results)
        self._worker.error.connect(self._show_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_worker_finished(self):
        self._worker = None
        self._refresh_model_state()

    def _show_results(self, results):
        self.search_btn.setEnabled(True)
        self.search_btn.setText("Search")
        self._clear_results()

        if not results:
            self.status_label.setText("No matching images found")
            empty = QLabel("No images found matching your query")
            empty.setObjectName("EmptyStateLabel")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_layout.addWidget(empty, 0, 0, 1, 5)
            return

        self.status_label.setText(f"Found {len(results)} image{'s' if len(results) != 1 else ''}")

        cols = 5
        for i, result in enumerate(results):
            card = ImageCard(result, self.app_core, thumb_size=170)
            self.results_layout.addWidget(card, i // cols, i % cols)

    def _show_error(self, msg):
        self._refresh_model_state()
        self.status_label.setText(f"Error: {msg}")

    def closeEvent(self, event):
        if self._model_state_timer.isActive():
            self._model_state_timer.stop()
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(2000)
        super().closeEvent(event)