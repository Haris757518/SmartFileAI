from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QLabel, QVBoxLayout, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer

from .sidebar import Sidebar
from .backend_loader import BackendLoader


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SmartFileAI")
        self.setMinimumSize(1260, 780)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        central_widget.setLayout(self.main_layout)

        self.sidebar = Sidebar()
        self.main_layout.addWidget(self.sidebar)

        # Placeholder while loading
        self.loading_widget = self._make_loading_widget()
        self.main_layout.addWidget(self.loading_widget)
        self.loading_error_widget = None

        self.pages = None
        self.statusBar().showMessage("  Initializing AI engine...")

        self.loader_thread = BackendLoader()
        self.loader_thread.finished_loading.connect(self._backend_ready)
        self.loader_thread.error_occurred.connect(self._backend_error)
        self.loader_thread.progress_changed.connect(self._on_loader_progress)
        self.loader_thread.start()

        self.loader_timeout_timer = QTimer(self)
        self.loader_timeout_timer.setSingleShot(True)
        self.loader_timeout_timer.timeout.connect(self._backend_timeout)
        self.loader_timeout_timer.start(120000)

    def _make_loading_widget(self):
        w = QWidget()
        w.setObjectName("LauncherPanel")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        w.setLayout(layout)

        icon = QLabel("◌")
        icon.setObjectName("LauncherGlyph")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)

        layout.addSpacing(16)

        title = QLabel("Loading AI Models")
        title.setObjectName("LauncherTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        self.loading_title = title

        subtitle = QLabel("E5 · SigLIP · FaceNet · FAISS")
        subtitle.setObjectName("LauncherSubtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        hint = QLabel("This may take 30–60 seconds on first launch")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setObjectName("LoadingLabel")
        layout.addWidget(hint)
        self.loading_hint = hint

        layout.addSpacing(12)

        progress = QProgressBar()
        progress.setObjectName("LauncherProgressBar")
        progress.setRange(0, 100)
        progress.setValue(5)
        progress.setFixedWidth(420)
        progress.setTextVisible(False)
        layout.addWidget(progress)
        self.loading_progress = progress

        progress_label = QLabel("5% - Starting...")
        progress_label.setObjectName("LauncherProgressText")
        progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(progress_label)
        self.loading_progress_label = progress_label

        return w

    def _on_loader_progress(self, percent, message):
        if hasattr(self, "loading_progress") and self.loading_progress:
            self.loading_progress.setValue(max(0, min(100, int(percent))))

        if hasattr(self, "loading_progress_label") and self.loading_progress_label:
            self.loading_progress_label.setText(f"{int(percent)}% - {message}")

        self.statusBar().showMessage(f"  {message}")

    def _backend_ready(self, app_core):
        from .stacked_pages import StackedPages

        if self.loader_timeout_timer.isActive():
            self.loader_timeout_timer.stop()

        self._on_loader_progress(100, "Ready")

        self.app_core = app_core

        # Remove loading widget
        self.main_layout.removeWidget(self.loading_widget)
        self.loading_widget.deleteLater()

        self.pages = StackedPages(self.app_core)
        self.main_layout.addWidget(self.pages)

        self._maybe_prompt_background_loading()

        self.sidebar.page_selected.connect(self._on_page_selected)
        self.sidebar.set_engine_ready(True)

        self.statusBar().showMessage("  ● SmartFileAI ready  —  GPU accelerated")

        # Show dashboard
        self.pages.switch_page("dashboard")

    def _maybe_prompt_background_loading(self):
        if not self.app_core.should_prompt_background_ai():
            if self.app_core.is_background_ai_enabled():
                self.app_core.start_background_image_preload(force=True)
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Enable Smart Background Loading?")
        box.setText("Enable Smart Background Loading?")
        box.setInformativeText(
            "This will make image search feel instant by preparing AI in the background.\n"
            "It uses some RAM while SmartFileAI is open."
        )

        enable_button = box.addButton("Enable", QMessageBox.ButtonRole.YesRole)
        box.addButton("No, load when needed", QMessageBox.ButtonRole.NoRole)
        box.exec()

        enabled = box.clickedButton() == enable_button
        self.app_core.set_background_ai(enabled)

    def _backend_error(self, error_msg):
        if self.loader_timeout_timer.isActive():
            self.loader_timeout_timer.stop()

        self.statusBar().showMessage(f"  ✗ Engine failed to load: {error_msg}")
        self.sidebar.set_engine_ready(False)

        if self.loading_widget:
            self.main_layout.removeWidget(self.loading_widget)
            self.loading_widget.deleteLater()
            self.loading_widget = None

        if self.loading_error_widget:
            self.main_layout.removeWidget(self.loading_error_widget)
            self.loading_error_widget.deleteLater()

        self.loading_error_widget = self._make_error_widget(error_msg)
        self.main_layout.addWidget(self.loading_error_widget)

    def _backend_timeout(self):
        if self.pages is not None:
            return

        if hasattr(self, "loading_title"):
            self.loading_title.setText("Still Initializing")
        if hasattr(self, "loading_hint"):
            self.loading_hint.setText(
                "Startup is taking longer than expected. If this continues, restart the app."
            )
        self.statusBar().showMessage("  Loading is taking longer than expected...")

    def _make_error_widget(self, error_msg):
        w = QWidget()
        w.setObjectName("LauncherPanel")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        w.setLayout(layout)

        title = QLabel("Engine Failed to Start")
        title.setObjectName("LauncherErrorTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        msg = QLabel(str(error_msg))
        msg.setObjectName("LauncherErrorText")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setWordWrap(True)
        layout.addWidget(msg)

        return w

    def _on_page_selected(self, key):
        if self.pages:
            self.pages.switch_page(key)

    def closeEvent(self, event):
        if hasattr(self, "app_core") and self.app_core:
            self.app_core.shutdown()
        super().closeEvent(event)