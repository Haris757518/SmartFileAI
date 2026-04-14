from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont


NAV_ITEMS = [
    ("⬡  Dashboard",   "dashboard"),
    ("⌕  Text Search", "text"),
    ("⊞  Image Search","image"),
    ("◉  Face Search", "face"),
    ("◎  Duplicates",  "duplicates"),
    ("⊕  Index Folder","index"),
    ("⚙  Settings",    "settings"),
]


class Sidebar(QWidget):
    page_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("Sidebar")
        self.setFixedWidth(210)

        self._buttons = {}
        self._active = "dashboard"

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 24, 12, 16)
        layout.setSpacing(0)
        self.setLayout(layout)

        # ---- Brand ----
        brand_widget = QWidget()
        brand_layout = QVBoxLayout()
        brand_layout.setContentsMargins(8, 0, 0, 0)
        brand_layout.setSpacing(2)
        brand_widget.setLayout(brand_layout)

        title = QLabel("SmartFileAI")
        title.setObjectName("AppTitle")
        brand_layout.addWidget(title)

        subtitle = QLabel("AI SEARCH ENGINE")
        subtitle.setObjectName("AppSubtitle")
        brand_layout.addWidget(subtitle)

        layout.addWidget(brand_widget)
        layout.addSpacing(24)

        # ---- Separator ----
        sep = QFrame()
        sep.setObjectName("Separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)
        layout.addSpacing(16)

        # ---- Nav buttons ----
        for label, key in NAV_ITEMS:
            btn = QPushButton(label)
            btn.setObjectName("SidebarButton")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, k=key: self._on_click(k))
            layout.addWidget(btn)
            layout.addSpacing(2)
            self._buttons[key] = btn

        layout.addStretch()

        # ---- Bottom status ----
        sep2 = QFrame()
        sep2.setObjectName("Separator")
        sep2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep2)
        layout.addSpacing(12)

        self.status_label = QLabel("● Offline")
        self.status_label.setObjectName("StatusDot")
        self.status_label.setStyleSheet("color: #FF8F8F; font-size: 11px; padding-left: 8px;")
        layout.addWidget(self.status_label)

        # Set dashboard active by default
        self._set_active("dashboard")

    def _on_click(self, key):
        self._set_active(key)
        self.page_selected.emit(key)

    def _set_active(self, key):
        if self._active and self._active in self._buttons:
            self._buttons[self._active].setObjectName("SidebarButton")
            self._buttons[self._active].style().unpolish(self._buttons[self._active])
            self._buttons[self._active].style().polish(self._buttons[self._active])

        self._active = key
        if key in self._buttons:
            self._buttons[key].setObjectName("SidebarButtonActive")
            self._buttons[key].style().unpolish(self._buttons[key])
            self._buttons[key].style().polish(self._buttons[key])

    def set_engine_ready(self, ready: bool):
        if ready:
            self.status_label.setText("● Engine Ready")
            self.status_label.setStyleSheet("color: #4CDDA3; font-size: 11px; padding-left: 8px;")
        else:
            self.status_label.setText("◌ Loading...")
            self.status_label.setStyleSheet("color: #F4B663; font-size: 11px; padding-left: 8px;")