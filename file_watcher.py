import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class SmartFileEventHandler(FileSystemEventHandler):

    def __init__(self, folder_path, app_core):
        self.folder_path = folder_path
        self.app = app_core

    def on_created(self, event):
        if event.is_directory:
            return
        print(f"\n[Watcher] File created: {event.src_path}")
        self.process(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        print(f"\n[Watcher] File modified: {event.src_path}")
        self.process(event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        print(f"\n[Watcher] File deleted: {event.src_path}")
        self.app.remove_file_from_index(event.src_path)

    def process(self, file_path):
        try:
            time.sleep(1.5)  # allow file write to finish
            print(f"[Watcher] Reindexing: {file_path}")
            self.app.reindex_single_file(file_path)
        except Exception as e:
            print(f"[Watcher Error] {e}")


def start_watcher(folder_path, app_core):
    event_handler = SmartFileEventHandler(folder_path, app_core)
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=True)
    observer.start()

    print(f"\nReal-time watcher started for: {folder_path}")

    return observer