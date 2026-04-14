import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog,
    QTextEdit, QFrame, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QTextCursor


# ======================================================
# INDEXING WORKER
# ======================================================
class IndexWorker(QThread):
    log_message = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    finished = pyqtSignal(bool, str)

    def __init__(self, folder_path, app_core):
        super().__init__()
        self.folder_path = folder_path
        self.app_core = app_core

    def run(self):
        try:
            import sys
            from image_indexer import ImageIndexer
            from indexer import FileIndexer

            self.log_message.emit(f"Starting indexing: {self.folder_path}")
            self.log_message.emit("━" * 50)

            # Text indexing
            self.log_message.emit("\n📄 Indexing text files (PDF, DOCX, TXT, PPTX)...")
            file_indexer = FileIndexer()

            text_count = 0
            for root, _, files in os.walk(self.folder_path):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in [".pdf", ".docx", ".txt", ".pptx", ".zip"]:
                        full_path = os.path.join(root, f)
                        try:
                            file_indexer.process_file(full_path)
                            text_count += 1
                            self.log_message.emit(f"  ✓ {f}")
                        except Exception as e:
                            self.log_message.emit(f"  ✗ {f}: {e}")

            self.log_message.emit(f"\nText files indexed: {text_count}")
            self.progress_update.emit(50)

            # Image indexing
            self.log_message.emit("\n🖼 Indexing images (face, gender, person detection)...")
            img_indexer = ImageIndexer()
            img_indexer.index_folder(self.folder_path)
            self.progress_update.emit(90)

            # Reload FAISS
            self.log_message.emit("\n⚡ Rebuilding search indexes...")
            self.app_core.refresh_all_indexes(force_rebuild=True)
            self.progress_update.emit(100)

            self.log_message.emit("\n━" * 50)
            self.log_message.emit("✅ Indexing complete! Search engine updated.")
            self.finished.emit(True, "Indexing completed successfully")

        except Exception as e:
            import traceback
            self.log_message.emit(f"\n✗ Error: {e}")
            self.log_message.emit(traceback.format_exc())
            self.finished.emit(False, str(e))


# ======================================================
# PAGE
# ======================================================
class IndexPage(QWidget):
    def __init__(self, app_core):
        super().__init__()
        self.app_core = app_core
        self._worker = None
        self._folder_path = None

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 36, 40, 24)
        layout.setSpacing(0)
        self.setLayout(layout)

        # Header
        title = QLabel("Index Folder")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel("Index a folder to make files searchable — text, images, and faces")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(28)

        # Folder selection
        folder_row = QHBoxLayout()
        folder_row.setSpacing(10)

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setObjectName("SearchInput")
        self.folder_label.setMinimumHeight(44)
        self.folder_label.setStyleSheet(
            "background-color: #111D29; border: 1px solid #233244; "
            "border-radius: 10px; padding: 12px 16px; color: #4D6780; font-size: 13px;"
        )
        folder_row.addWidget(self.folder_label, 1)

        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("SecondaryButton")
        browse_btn.setMinimumHeight(44)
        browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(browse_btn)

        self.index_btn = QPushButton("⊕  Start Indexing")
        self.index_btn.setObjectName("PrimaryButton")
        self.index_btn.setMinimumHeight(44)
        self.index_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.index_btn.clicked.connect(self._start_indexing)
        self.index_btn.setEnabled(False)
        folder_row.addWidget(self.index_btn)

        layout.addLayout(folder_row)
        layout.addSpacing(16)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximumHeight(6)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Status
        self.status_label = QLabel("")
        self.status_label.setObjectName("LoadingLabel")
        layout.addWidget(self.status_label)

        layout.addSpacing(16)

        # Log area label
        log_header = QLabel("INDEXING LOG")
        log_header.setObjectName("SettingSection")
        layout.addWidget(log_header)

        layout.addSpacing(10)

        # Log text area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet(
            "background-color: #091018; color: #5F7D9A; "
            "font-family: 'Consolas', monospace; font-size: 12px; "
            "border: 1px solid #152231; border-radius: 8px; padding: 12px;"
        )
        self.log_area.setPlaceholderText(
            "Indexing output will appear here…\n\n"
            "Select a folder and click Start Indexing."
        )
        layout.addWidget(self.log_area, 1)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Index")
        if folder:
            self._folder_path = folder
            self.folder_label.setText(folder)
            self.folder_label.setStyleSheet(
                "background-color: #111D29; border: 1px solid #233244; "
                "border-radius: 10px; padding: 12px 16px; color: #E6F1FF; font-size: 13px;"
            )
            self.index_btn.setEnabled(True)
            self.status_label.setText("Folder selected — ready to index")

    def _start_indexing(self):
        if not self._folder_path:
            return

        self.index_btn.setEnabled(False)
        self.index_btn.setText("Indexing…")
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.log_area.clear()
        self.status_label.setText("Indexing in progress…")

        self._worker = IndexWorker(self._folder_path, self.app_core)
        self._worker.log_message.connect(self._append_log)
        self._worker.progress_update.connect(self.progress_bar.setValue)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, msg):
        self.log_area.append(msg)
        self.log_area.moveCursor(QTextCursor.MoveOperation.End)

    def _on_finished(self, success, msg):
        self.index_btn.setEnabled(True)
        self.index_btn.setText("⊕  Start Indexing")

        if success:
            self.status_label.setText("✅ Indexing complete — search engine updated")
            self.status_label.setStyleSheet("color: #4CDDA3; font-size: 13px;")
        else:
            self.status_label.setText(f"✗ Error: {msg}")
            self.status_label.setStyleSheet("color: #FF8F8F; font-size: 13px;")