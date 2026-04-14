import os
import shutil
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog,
    QFrame, QScrollArea, QSpinBox, QComboBox,
    QCheckBox, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal


class DuplicateScanWorker(QThread):
    results_ready = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, app_core, folder_path, mode, threshold):
        super().__init__()
        self.app_core = app_core
        self.folder_path = folder_path
        self.mode = mode
        self.threshold = threshold

    def run(self):
        try:
            results = self.app_core.find_duplicate_images(
                self.folder_path,
                mode=self.mode,
                phash_threshold=self.threshold,
            )
            self.results_ready.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class DuplicateGroupCard(QWidget):
    def __init__(self, group_title, files, app_core, selection_changed_callback):
        super().__init__()
        self.app_core = app_core
        self._rows = []
        self.setObjectName("ResultCard")

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        self.setLayout(layout)

        title = QLabel(group_title)
        title.setObjectName("ResultFileName")
        layout.addWidget(title)

        for idx, path in enumerate(files):
            row = QHBoxLayout()
            row.setSpacing(8)

            check = QCheckBox()
            check.setCursor(Qt.CursorShape.PointingHandCursor)
            check.stateChanged.connect(selection_changed_callback)
            row.addWidget(check)

            if idx == 0:
                original = QLabel("Original")
                original.setObjectName("ScoreBadge")
                row.addWidget(original)

            label = QLabel(path)
            label.setObjectName("ResultSnippet")
            label.setWordWrap(True)
            row.addWidget(label, 1)

            open_btn = QPushButton("Open")
            open_btn.setObjectName("IconButton")
            open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            open_btn.clicked.connect(lambda _, p=path: self.app_core.open_file(p))
            row.addWidget(open_btn)

            layout.addLayout(row)
            self._rows.append((check, path))

    def clear_selection(self):
        for check, _ in self._rows:
            check.blockSignals(True)
            check.setChecked(False)
            check.blockSignals(False)

    def select_duplicates_only(self):
        for idx, (check, _) in enumerate(self._rows):
            check.blockSignals(True)
            check.setChecked(idx > 0)
            check.blockSignals(False)

    def selected_paths(self):
        return [path for check, path in self._rows if check.isChecked()]


class DuplicateSearchPage(QWidget):
    def __init__(self, app_core):
        super().__init__()
        self.app_core = app_core
        self._worker = None
        self._folder_path = None
        self._group_cards = []

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 36, 40, 20)
        layout.setSpacing(0)
        self.setLayout(layout)

        title = QLabel("Duplicate Finder")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel("Find exact or visually similar duplicate images in a folder")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(24)

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

        layout.addLayout(folder_row)
        layout.addSpacing(12)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Exact Duplicates", "exact")
        self.mode_combo.addItem("Similar Images", "similar")
        self.mode_combo.addItem("Hybrid (Exact + Similar)", "hybrid")
        self.mode_combo.setMinimumHeight(40)
        controls_row.addWidget(self.mode_combo)

        threshold_label = QLabel("Similarity Threshold")
        threshold_label.setObjectName("LoadingLabel")
        controls_row.addWidget(threshold_label)

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 20)
        self.threshold_spin.setValue(8)
        self.threshold_spin.setMinimumHeight(40)
        self.threshold_spin.setFixedWidth(88)
        controls_row.addWidget(self.threshold_spin)

        controls_row.addStretch()

        self.scan_btn = QPushButton("Run Scan")
        self.scan_btn.setObjectName("PrimaryButton")
        self.scan_btn.setMinimumHeight(42)
        self.scan_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.scan_btn.setEnabled(False)
        self.scan_btn.clicked.connect(self._run_scan)
        controls_row.addWidget(self.scan_btn)

        layout.addLayout(controls_row)
        layout.addSpacing(10)

        self.status_label = QLabel("")
        self.status_label.setObjectName("LoadingLabel")
        layout.addWidget(self.status_label)

        layout.addSpacing(8)

        selection_row = QHBoxLayout()
        selection_row.setSpacing(8)

        self.select_all_btn = QPushButton("Select All Duplicates")
        self.select_all_btn.setObjectName("SecondaryButton")
        self.select_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.select_all_btn.clicked.connect(self._select_all_duplicates)
        self.select_all_btn.setEnabled(False)
        selection_row.addWidget(self.select_all_btn)

        self.clear_selection_btn = QPushButton("Clear Selection")
        self.clear_selection_btn.setObjectName("SecondaryButton")
        self.clear_selection_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_selection_btn.clicked.connect(self._clear_selection)
        self.clear_selection_btn.setEnabled(False)
        selection_row.addWidget(self.clear_selection_btn)

        self.move_selected_btn = QPushButton("Move Selected")
        self.move_selected_btn.setObjectName("PrimaryButton")
        self.move_selected_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.move_selected_btn.clicked.connect(self._move_selected)
        self.move_selected_btn.setEnabled(False)
        selection_row.addWidget(self.move_selected_btn)

        self.delete_selected_btn = QPushButton("Delete Selected")
        self.delete_selected_btn.setObjectName("DangerButton")
        self.delete_selected_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_selected_btn.clicked.connect(self._delete_selected)
        self.delete_selected_btn.setEnabled(False)
        selection_row.addWidget(self.delete_selected_btn)

        selection_row.addStretch()
        layout.addLayout(selection_row)

        layout.addSpacing(8)

        self.selection_label = QLabel("Selected 0 files")
        self.selection_label.setObjectName("LoadingLabel")
        layout.addWidget(self.selection_label)

        layout.addSpacing(8)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.results_container = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(8)
        self.results_container.setLayout(self.results_layout)
        self.scroll.setWidget(self.results_container)
        layout.addWidget(self.scroll, 1)

        self._show_empty_state()

    def _show_empty_state(self):
        self._clear_results()
        self._group_cards = []
        self._update_selection_state()
        self.select_all_btn.setEnabled(False)
        empty = QLabel("Select a folder and run a duplicate scan")
        empty.setObjectName("EmptyStateLabel")
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(empty)
        self.results_layout.addStretch()

    def _clear_results(self):
        while self.results_layout.count() > 0:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self._folder_path = folder
            self.folder_label.setText(folder)
            self.folder_label.setStyleSheet(
                "background-color: #111D29; border: 1px solid #233244; "
                "border-radius: 10px; padding: 12px 16px; color: #E6F1FF; font-size: 13px;"
            )
            self.scan_btn.setEnabled(True)
            self.status_label.setText("Folder selected. Ready to scan.")

    def _run_scan(self):
        if not self._folder_path:
            return

        if self._worker and self._worker.isRunning():
            return

        mode = self.mode_combo.currentData()
        threshold = int(self.threshold_spin.value())

        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("Scanning...")
        self.status_label.setText("Scanning images for duplicates...")
        self._clear_results()
        self._group_cards = []
        self._update_selection_state()
        self.select_all_btn.setEnabled(False)

        self._worker = DuplicateScanWorker(
            self.app_core,
            self._folder_path,
            mode,
            threshold,
        )
        self._worker.results_ready.connect(self._show_results)
        self._worker.error.connect(self._show_error)
        self._worker.finished.connect(self._on_scan_finished)
        self._worker.start()

    def _add_section(self, title_text, groups):
        section = QLabel(title_text)
        section.setObjectName("SettingSection")
        self.results_layout.addWidget(section)

        for idx, files in enumerate(groups, start=1):
            card = DuplicateGroupCard(
                f"Group {idx}  |  {len(files)} files",
                files,
                self.app_core,
                self._update_selection_state,
            )
            self.results_layout.addWidget(card)
            self._group_cards.append(card)

        self.results_layout.addSpacing(8)

    def _selected_paths(self):
        selected = set()
        for card in self._group_cards:
            for path in card.selected_paths():
                selected.add(path)
        return sorted(selected)

    def _update_selection_state(self):
        count = len(self._selected_paths())
        self.selection_label.setText(f"Selected {count} file{'s' if count != 1 else ''}")
        self.clear_selection_btn.setEnabled(count > 0)
        self.move_selected_btn.setEnabled(count > 0)
        self.delete_selected_btn.setEnabled(count > 0)

    def _select_all_duplicates(self):
        for card in self._group_cards:
            card.select_duplicates_only()
        self._update_selection_state()

    def _clear_selection(self):
        for card in self._group_cards:
            card.clear_selection()
        self._update_selection_state()

    def _build_destination_path(self, destination_folder, file_name):
        base_name, ext = os.path.splitext(file_name)
        candidate = os.path.join(destination_folder, file_name)
        counter = 1

        while os.path.exists(candidate):
            candidate = os.path.join(destination_folder, f"{base_name}_{counter}{ext}")
            counter += 1

        return candidate

    def _move_selected(self):
        paths = self._selected_paths()
        if not paths:
            return

        destination = QFileDialog.getExistingDirectory(self, "Select destination folder")
        if not destination:
            return

        reply = QMessageBox.question(
            self,
            "Move Selected Files",
            f"Move {len(paths)} selected file(s) to:\n{destination}\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        moved = 0
        failed = 0
        for path in paths:
            if not os.path.isfile(path):
                failed += 1
                continue

            try:
                target_path = self._build_destination_path(destination, os.path.basename(path))
                shutil.move(path, target_path)
                moved += 1
            except Exception:
                failed += 1

        self.status_label.setText(f"Moved {moved} file(s). Failed: {failed}")
        self._run_scan()

    def _delete_selected(self):
        paths = self._selected_paths()
        if not paths:
            return

        reply = QMessageBox.warning(
            self,
            "Delete Selected Files",
            f"Delete {len(paths)} selected file(s) permanently?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        deleted = 0
        failed = 0
        for path in paths:
            if not os.path.isfile(path):
                failed += 1
                continue

            try:
                os.remove(path)
                deleted += 1
            except Exception:
                failed += 1

        self.status_label.setText(f"Deleted {deleted} file(s). Failed: {failed}")
        self._run_scan()

    def _show_results(self, results):
        self._clear_results()
        self._group_cards = []

        total_images = results.get("total_images", 0)
        exact_groups = results.get("exact_groups", [])
        similar_groups = results.get("similar_groups", [])

        total_groups = len(exact_groups) + len(similar_groups)
        if total_groups == 0:
            self.status_label.setText(f"No duplicates found in {total_images} images")
            empty = QLabel("No duplicate groups detected")
            empty.setObjectName("EmptyStateLabel")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_layout.addWidget(empty)
            self.results_layout.addStretch()
            self.select_all_btn.setEnabled(False)
            self._update_selection_state()
            return

        self.status_label.setText(
            f"Scanned {total_images} images  |  Found {total_groups} duplicate groups"
        )
        self.select_all_btn.setEnabled(True)

        if exact_groups:
            self._add_section(f"EXACT DUPLICATES ({len(exact_groups)} groups)", exact_groups)

        if similar_groups:
            self._add_section(f"SIMILAR IMAGES ({len(similar_groups)} groups)", similar_groups)

        self.results_layout.addStretch()
        self._update_selection_state()

    def _show_error(self, msg):
        self.status_label.setText(f"Error: {msg}")
        self._show_empty_state()

    def _on_scan_finished(self):
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Run Scan")
        self._worker = None
