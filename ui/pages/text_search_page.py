import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit,
    QScrollArea, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut


# ======================================================
# SEARCH WORKER
# ======================================================
class TextSearchWorker(QThread):
    results_ready = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, app_core, query):
        super().__init__()
        self.app_core = app_core
        self.query = query

    def run(self):
        try:
            results = self.app_core.search_text(self.query, top_k=10)
            self.results_ready.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ======================================================
# RESULT CARD WIDGET
# ======================================================
class TextResultCard(QWidget):
    def __init__(self, result, app_core):
        super().__init__()
        self.app_core = app_core
        self.file_path = result["file_path"]
        self.setObjectName("ResultCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout()
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(6)
        self.setLayout(layout)

        # Top row: filename + score + open button
        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        # File type badge
        ext = os.path.splitext(result["file_name"])[1].upper().lstrip(".")
        badge = QLabel(ext or "FILE")
        badge.setObjectName("FileTypeBadge")
        badge.setFixedWidth(42)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_row.addWidget(badge)

        # File name
        name_label = QLabel(result["file_name"])
        name_label.setObjectName("ResultFileName")
        name_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        top_row.addWidget(name_label, 1)

        # Score badge
        score_val = result.get("final_score", 0)
        score_label = QLabel(f"Score {score_val:.2f}")
        score_label.setObjectName("ScoreBadge")
        top_row.addWidget(score_label)

        # Open button
        open_btn = QPushButton("Open ↗")
        open_btn.setObjectName("IconButton")
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.clicked.connect(lambda: self.app_core.open_file(self.file_path))
        top_row.addWidget(open_btn)

        layout.addLayout(top_row)

        # Snippet
        snippet = result.get("snippet", "")
        if snippet:
            snippet_label = QLabel(snippet[:280] + ("…" if len(snippet) > 280 else ""))
            snippet_label.setObjectName("ResultSnippet")
            snippet_label.setWordWrap(True)
            layout.addWidget(snippet_label)

        # Path (truncated)
        path_label = QLabel(self.file_path)
        path_label.setStyleSheet("color: #2B445B; font-size: 10px;")
        path_label.setWordWrap(True)
        layout.addWidget(path_label)

    def mouseDoubleClickEvent(self, event):
        self.app_core.open_file(self.file_path)


# ======================================================
# PAGE
# ======================================================
class TextSearchPage(QWidget):
    def __init__(self, app_core):
        super().__init__()
        self.app_core = app_core
        self._worker = None

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 36, 40, 20)
        layout.setSpacing(0)
        self.setLayout(layout)

        # Header
        title = QLabel("Text Search")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel("Semantic search across PDF · DOCX · TXT · PPTX · ZIP")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(24)

        # Search bar row
        search_row = QHBoxLayout()
        search_row.setSpacing(10)

        self.search_input = QLineEdit()
        self.search_input.setObjectName("SearchInput")
        self.search_input.setPlaceholderText("Search documents, reports, notes…")
        self.search_input.setMinimumHeight(44)
        self.search_input.returnPressed.connect(self._run_search)
        search_row.addWidget(self.search_input, 1)

        self.search_btn = QPushButton("Search")
        self.search_btn.setObjectName("PrimaryButton")
        self.search_btn.setMinimumHeight(44)
        self.search_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.search_btn.clicked.connect(self._run_search)
        search_row.addWidget(self.search_btn)

        layout.addLayout(search_row)
        layout.addSpacing(20)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setObjectName("LoadingLabel")
        layout.addWidget(self.status_label)

        layout.addSpacing(8)

        # Scroll area for results
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.results_container = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(8)
        self.results_layout.addStretch()
        self.results_container.setLayout(self.results_layout)
        self.scroll.setWidget(self.results_container)
        layout.addWidget(self.scroll, 1)

        self._show_empty_state()

    def _show_empty_state(self):
        self._clear_results()
        empty = QLabel("Type a query and press Search")
        empty.setObjectName("EmptyStateLabel")
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_layout.insertWidget(0, empty)

    def _clear_results(self):
        while self.results_layout.count() > 0:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _run_search(self):
        query = self.search_input.text().strip()
        if not query:
            return

        self.search_btn.setEnabled(False)
        self.search_btn.setText("Searching…")
        self.status_label.setText("Searching…")
        self._clear_results()

        self._worker = TextSearchWorker(self.app_core, query)
        self._worker.results_ready.connect(self._show_results)
        self._worker.error.connect(self._show_error)
        self._worker.start()

    def _show_results(self, results):
        self.search_btn.setEnabled(True)
        self.search_btn.setText("Search")
        self._clear_results()

        if not results:
            self.status_label.setText("No results found")
            empty = QLabel("No matching documents found for this query")
            empty.setObjectName("EmptyStateLabel")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_layout.addWidget(empty)
            self.results_layout.addStretch()
            return

        self.status_label.setText(f"Found {len(results)} result{'s' if len(results) != 1 else ''}")

        for result in results:
            card = TextResultCard(result, self.app_core)
            self.results_layout.addWidget(card)

        self.results_layout.addStretch()

    def _show_error(self, msg):
        self.search_btn.setEnabled(True)
        self.search_btn.setText("Search")
        self.status_label.setText(f"Error: {msg}")