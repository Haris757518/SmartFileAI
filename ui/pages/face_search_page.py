import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog,
    QScrollArea, QFrame, QGridLayout,
    QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap


# ======================================================
# WORKER
# ======================================================
class FaceSearchWorker(QThread):
    results_ready = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, app_core, image_path):
        super().__init__()
        self.app_core = app_core
        self.image_path = image_path

    def run(self):
        try:
            results = self.app_core.search_by_face(self.image_path, top_k=20)
            self.results_ready.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ======================================================
# FACE RESULT CARD
# ======================================================
class FaceResultCard(QWidget):
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
                img_label.setText("👤")
                img_label.setStyleSheet("font-size: 32px; color: #3C556D; background-color: #0B1118;")
        else:
            img_label.setText("✗")

        layout.addWidget(img_label)

        bottom = QHBoxLayout()
        score_val = result.get("final_score", 0)

        # Color the score by confidence
        if score_val >= 0.85:
            color = "#4CDDA3"
        elif score_val >= 0.75:
            color = "#F4B663"
        else:
            color = "#FF8F8F"

        score = QLabel(f"{score_val:.2f}")
        score.setStyleSheet(
            f"background-color: #152231; color: {color}; "
            f"border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 600;"
        )
        bottom.addWidget(score)

        name = QLabel(result.get("file_name", os.path.basename(self.file_path))[:22])
        name.setObjectName("ImageLabel")
        name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bottom.addWidget(name, 1)

        layout.addLayout(bottom)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.app_core.open_file(self.file_path)


# ======================================================
# PAGE
# ======================================================
class FaceSearchPage(QWidget):
    def __init__(self, app_core):
        super().__init__()
        self.app_core = app_core
        self._worker = None
        self._selected_image = None

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 36, 40, 20)
        layout.setSpacing(0)
        self.setLayout(layout)

        # Header
        title = QLabel("Face Search")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel("Upload an image with a face to find matching photos")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(24)

        # Upload area
        upload_row = QHBoxLayout()
        upload_row.setSpacing(12)

        self.upload_btn = QPushButton("◉  Upload Face Image")
        self.upload_btn.setObjectName("PrimaryButton")
        self.upload_btn.setMinimumHeight(44)
        self.upload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.upload_btn.clicked.connect(self._upload_image)
        upload_row.addWidget(self.upload_btn)

        self.search_btn = QPushButton("Search Faces")
        self.search_btn.setObjectName("SecondaryButton")
        self.search_btn.setMinimumHeight(44)
        self.search_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.search_btn.clicked.connect(self._run_search)
        self.search_btn.setEnabled(False)
        upload_row.addWidget(self.search_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("SecondaryButton")
        self.clear_btn.setMinimumHeight(44)
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.clicked.connect(self._clear)
        self.clear_btn.hide()
        upload_row.addWidget(self.clear_btn)

        upload_row.addStretch()
        layout.addLayout(upload_row)

        layout.addSpacing(12)

        # Preview + info row
        self.preview_row = QHBoxLayout()
        self.preview_row.setSpacing(16)

        self.preview_widget = QLabel()
        self.preview_widget.setFixedSize(100, 100)
        self.preview_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_widget.setStyleSheet(
            "border-radius: 8px; background-color: #101923; border: 1px solid #1C2A3A;"
        )
        self.preview_widget.setText("No image")
        self.preview_widget.setObjectName("LoadingLabel")
        self.preview_row.addWidget(self.preview_widget)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        self.image_name_label = QLabel("No image selected")
        self.image_name_label.setObjectName("ResultFileName")
        info_layout.addWidget(self.image_name_label)

        self.hint_label = QLabel(
            "Upload a photo with a visible face.\n"
            "The engine will search for matching faces using FaceNet embeddings."
        )
        self.hint_label.setObjectName("ResultSnippet")
        self.hint_label.setWordWrap(True)
        info_layout.addWidget(self.hint_label)

        info_layout.addStretch()
        self.preview_row.addLayout(info_layout, 1)

        layout.addLayout(self.preview_row)

        layout.addSpacing(8)

        # Status
        self.status_label = QLabel("")
        self.status_label.setObjectName("LoadingLabel")
        layout.addWidget(self.status_label)

        layout.addSpacing(8)

        # Results
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

    def _show_empty_state(self):
        self._clear_results()
        empty = QLabel("Upload an image to begin face search")
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
            self, "Select Image with Face", "",
            "Images (*.jpg *.jpeg *.png *.webp *.bmp)"
        )
        if path:
            self._selected_image = path
            pix = QPixmap(path).scaled(
                96, 96,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_widget.setPixmap(pix)
            self.image_name_label.setText(os.path.basename(path))
            self.search_btn.setEnabled(True)
            self.clear_btn.show()
            self.status_label.setText("Image ready — click Search Faces")

    def _clear(self):
        self._selected_image = None
        self.preview_widget.clear()
        self.preview_widget.setText("No image")
        self.image_name_label.setText("No image selected")
        self.search_btn.setEnabled(False)
        self.clear_btn.hide()
        self.status_label.setText("")
        self._show_empty_state()

    def _run_search(self):
        if not self._selected_image:
            return

        self.search_btn.setEnabled(False)
        self.search_btn.setText("Searching…")
        self.status_label.setText("Detecting and matching faces…")
        self._clear_results()

        self._worker = FaceSearchWorker(self.app_core, self._selected_image)
        self._worker.results_ready.connect(self._show_results)
        self._worker.error.connect(self._show_error)
        self._worker.start()

    def _show_results(self, results):
        self.search_btn.setEnabled(True)
        self.search_btn.setText("Search Faces")
        self._clear_results()

        if not results:
            self.status_label.setText("No matching faces found")
            empty = QLabel("No faces matched — try a clearer photo with a visible face")
            empty.setObjectName("EmptyStateLabel")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_layout.addWidget(empty, 0, 0, 1, 5)
            return

        self.status_label.setText(
            f"Found {len(results)} match{'es' if len(results) != 1 else ''}  "
            f"(green = high confidence, orange = medium, red = low)"
        )

        cols = 5
        for i, result in enumerate(results):
            card = FaceResultCard(result, self.app_core)
            self.results_layout.addWidget(card, i // cols, i % cols)

    def _show_error(self, msg):
        self.search_btn.setEnabled(True)
        self.search_btn.setText("Search Faces")
        self.status_label.setText(f"Error: {msg}")