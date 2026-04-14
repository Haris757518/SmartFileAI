from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame,
    QMessageBox, QApplication, QCheckBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt


class SettingRow(QWidget):
    def __init__(self, label, description, widget=None):
        super().__init__()

        layout = QHBoxLayout()
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(16)
        self.setLayout(layout)
        self.setObjectName("ResultCard")

        text_layout = QVBoxLayout()
        text_layout.setSpacing(3)

        lbl = QLabel(label)
        lbl.setObjectName("SettingLabel")
        text_layout.addWidget(lbl)

        desc = QLabel(description)
        desc.setObjectName("SettingDescription")
        text_layout.addWidget(desc)

        layout.addLayout(text_layout, 1)

        if widget:
            layout.addWidget(widget)


class SettingsPage(QWidget):
    def __init__(self, app_core):
        super().__init__()
        self.app_core = app_core

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 36, 40, 36)
        layout.setSpacing(0)
        self.setLayout(layout)

        # Header
        title = QLabel("Settings")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel("Configure SmartFileAI")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(32)

        # ---- Engine section ----
        section1 = QLabel("ENGINE")
        section1.setObjectName("SettingSection")
        layout.addWidget(section1)
        layout.addSpacing(10)

        # GPU status
        gpu_btn = QLabel()
        gpu_btn.setText("Auto (GPU if available)")
        gpu_btn.setStyleSheet(
            "color: #4CDDA3; background-color: #163028; "
            "border-radius: 6px; padding: 6px 14px; font-size: 12px; font-weight: 600;"
        )

        layout.addWidget(SettingRow(
            "Compute Device",
            "AI models run on this device. GPU is strongly recommended.",
            gpu_btn
        ))
        layout.addSpacing(6)

        # Models status
        models_label = QLabel("E5-base · SigLIP · FaceNet · FasterRCNN")
        models_label.setStyleSheet("color: #5F7D9A; font-size: 12px; padding: 6px 14px;")
        layout.addWidget(SettingRow(
            "Loaded Models",
            "All AI models are active and running.",
            models_label
        ))
        layout.addSpacing(6)

        self.background_ai_toggle = QCheckBox("Enable")
        self.background_ai_toggle.setChecked(self.app_core.is_background_ai_enabled())
        self.background_ai_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.background_ai_toggle.stateChanged.connect(self._toggle_background_ai)
        layout.addWidget(SettingRow(
            "Background AI Loading",
            "Preload image AI in background for faster first search.",
            self.background_ai_toggle
        ))
        layout.addSpacing(6)

        self.image_threshold_spin = QDoubleSpinBox()
        self.image_threshold_spin.setRange(0.0, 1.0)
        self.image_threshold_spin.setDecimals(2)
        self.image_threshold_spin.setSingleStep(0.01)
        self.image_threshold_spin.setFixedWidth(100)
        self.image_threshold_spin.setValue(self.app_core.get_image_text_min_score())
        self.image_threshold_spin.valueChanged.connect(self._set_image_threshold)
        layout.addWidget(SettingRow(
            "Image Relevance Threshold",
            "Hide low-confidence image text matches. Higher values are stricter.",
            self.image_threshold_spin
        ))
        layout.addSpacing(24)

        # ---- Database section ----
        sep = QFrame()
        sep.setObjectName("Separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)
        layout.addSpacing(20)

        section2 = QLabel("DATABASE")
        section2.setObjectName("SettingSection")
        layout.addWidget(section2)
        layout.addSpacing(10)

        # DB path
        db_info = QLabel("data/index.db")
        db_info.setStyleSheet("color: #5F7D9A; font-size: 12px; font-family: Consolas; padding: 6px 14px;")
        layout.addWidget(SettingRow(
            "Index Database",
            "SQLite database storing all embeddings and metadata.",
            db_info
        ))
        layout.addSpacing(6)

        # Clear text index
        clear_text_btn = QPushButton("Clear Text Index")
        clear_text_btn.setObjectName("DangerButton")
        clear_text_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_text_btn.clicked.connect(self._clear_text_index)
        layout.addWidget(SettingRow(
            "Clear Text Index",
            "Remove all indexed text chunks. Does not delete images or faces.",
            clear_text_btn
        ))
        layout.addSpacing(24)

        # ---- About section ----
        sep2 = QFrame()
        sep2.setObjectName("Separator")
        sep2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep2)
        layout.addSpacing(20)

        section3 = QLabel("ABOUT")
        section3.setObjectName("SettingSection")
        layout.addWidget(section3)
        layout.addSpacing(10)

        ver_label = QLabel("1.0.0")
        ver_label.setStyleSheet("color: #5F7D9A; font-size: 12px; padding: 6px 14px;")
        layout.addWidget(SettingRow("SmartFileAI", "AI-powered local file search engine", ver_label))
        layout.addSpacing(6)

        stack_label = QLabel("PyTorch · FAISS · E5 · SigLIP · FaceNet · PyQt6")
        stack_label.setStyleSheet("color: #3C556D; font-size: 11px; padding: 6px 14px;")
        layout.addWidget(SettingRow("Tech Stack", "Core libraries powering the search engine.", stack_label))

        layout.addStretch()

    def _clear_text_index(self):
        reply = QMessageBox.question(
            self,
            "Clear Text Index",
            "This will remove all indexed text chunks from the database.\n\n"
            "Your images and face data will NOT be affected.\n\n"
            "Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.app_core.db.clear_text_index()
                self.app_core.load_all_text_data(force_rebuild=True)
                QMessageBox.information(self, "Done", "Text index cleared successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clear index:\n{e}")

    def _toggle_background_ai(self, state):
        enabled = state == int(Qt.CheckState.Checked)
        self.app_core.set_background_ai(enabled)

    def _set_image_threshold(self, value):
        self.app_core.set_image_text_min_score(value)