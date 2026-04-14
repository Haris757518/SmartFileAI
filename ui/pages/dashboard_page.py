from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QGridLayout
)
from PyQt6.QtCore import Qt


class StatCard(QWidget):
    def __init__(self, number, label, icon):
        super().__init__()
        self.setObjectName("StatCard")
        self.setMinimumHeight(110)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(4)
        self.setLayout(layout)

        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 20px;")
        layout.addWidget(icon_label)

        self.num_label = QLabel(str(number))
        self.num_label.setObjectName("StatNumber")
        layout.addWidget(self.num_label)

        lbl = QLabel(label.upper())
        lbl.setObjectName("StatLabel")
        layout.addWidget(lbl)

    def update_value(self, value):
        self.num_label.setText(str(value))


class DashboardPage(QWidget):
    def __init__(self, app_core):
        super().__init__()
        self.app_core = app_core

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 36, 40, 36)
        layout.setSpacing(0)
        self.setLayout(layout)

        # Header
        title = QLabel("Dashboard")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel("Overview of your indexed knowledge base")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(32)

        # Stats row
        stats = app_core.get_stats()

        grid = QGridLayout()
        grid.setSpacing(16)

        self.stat_cards = {
            "text":   StatCard(stats["text_chunks"],  "Text Chunks",  "📄"),
            "images": StatCard(stats["images"],        "Images",       "🖼"),
            "faces":  StatCard(stats["faces"],         "Faces",        "👤"),
            "persons":StatCard(stats["persons"],       "Persons",      "👥"),
        }

        grid.addWidget(self.stat_cards["text"],    0, 0)
        grid.addWidget(self.stat_cards["images"],  0, 1)
        grid.addWidget(self.stat_cards["faces"],   0, 2)
        grid.addWidget(self.stat_cards["persons"], 0, 3)

        layout.addLayout(grid)

        layout.addSpacing(32)

        # Separator
        sep = QFrame()
        sep.setObjectName("Separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        layout.addSpacing(24)

        # Feature overview
        features_title = QLabel("CAPABILITIES")
        features_title.setObjectName("SettingSection")
        layout.addWidget(features_title)

        layout.addSpacing(16)

        features = [
            ("⌕  Text Search",     "Semantic + lexical search across PDF, DOCX, TXT, PPTX files"),
            ("⊞  Image Search",    "Search images by text description or visual similarity (SigLIP)"),
            ("◉  Face Search",     "Find matching faces across your entire image library (FaceNet)"),
            ("◎  Duplicates",      "Detect exact and visually similar duplicate images in one scan"),
            ("⊕  Index Folder",    "Index a folder and watch it for changes in real-time"),
        ]

        for feat_title, feat_desc in features:
            row = QWidget()
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(16, 12, 16, 12)
            row_layout.setSpacing(16)
            row.setLayout(row_layout)
            row.setObjectName("ResultCard")

            title_lbl = QLabel(feat_title)
            title_lbl.setObjectName("ResultFileName")
            title_lbl.setMinimumWidth(160)
            row_layout.addWidget(title_lbl)

            desc_lbl = QLabel(feat_desc)
            desc_lbl.setObjectName("ResultSnippet")
            row_layout.addWidget(desc_lbl, 1)

            layout.addWidget(row)
            layout.addSpacing(6)

        layout.addStretch()

        # Footer
        footer = QLabel("SmartFileAI  ·  GPU Accelerated  ·  Local & Private")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("color: #2B445B; font-size: 11px;")
        layout.addWidget(footer)