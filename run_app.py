"""
SmartFileAI — Entry Point
Run: python run_app.py
"""

import sys
import os
import multiprocessing


def configure_model_cache():
    base_dir = os.path.dirname(__file__)
    default_cache = os.path.join(base_dir, "models", "hf_cache")

    legacy_cache = os.environ.get("TRANSFORMERS_CACHE")
    if legacy_cache and "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = legacy_cache

    os.environ.setdefault("HF_HOME", default_cache)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)


configure_model_cache()

# Required for PyInstaller + multiprocessing on Windows
if __name__ == "__main__":
    multiprocessing.freeze_support()

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from ui.main_window import MainWindow


def load_stylesheet(app):
    qss_path = os.path.join(os.path.dirname(__file__), "ui", "theme", "dark.qss")
    try:
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        print(f"[Warning] Theme file not found: {qss_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("SmartFileAI")

    load_stylesheet(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())