from PyQt6.QtCore import QThread, pyqtSignal


class BackendLoader(QThread):
    finished_loading = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    progress_changed = pyqtSignal(int, str)

    def _emit_progress(self, percent, message):
        self.progress_changed.emit(percent, message)

    def run(self):
        try:
            from app_core import AppCore
            app_core = AppCore(progress_callback=self._emit_progress)
            self.finished_loading.emit(app_core)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))