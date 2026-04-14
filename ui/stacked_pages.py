from PyQt6.QtWidgets import QStackedWidget

from .pages.dashboard_page import DashboardPage
from .pages.text_search_page import TextSearchPage
from .pages.image_search_page import ImageSearchPage
from .pages.face_search_page import FaceSearchPage
from .pages.duplicate_search_page import DuplicateSearchPage
from .pages.index_page import IndexPage
from .pages.settings_page import SettingsPage


class StackedPages(QStackedWidget):
    def __init__(self, app_core):
        super().__init__()
        self.app_core = app_core
        self.pages = {}

        self._page_factories = {
            "dashboard": lambda: DashboardPage(self.app_core),
            "text":      lambda: TextSearchPage(self.app_core),
            "image":     lambda: ImageSearchPage(self.app_core),
            "face":      lambda: FaceSearchPage(self.app_core),
            "duplicates":lambda: DuplicateSearchPage(self.app_core),
            "index":     lambda: IndexPage(self.app_core),
            "settings":  lambda: SettingsPage(self.app_core),
        }

        # Render dashboard instantly and lazily build other pages on demand.
        self._ensure_page("dashboard")
        self.setCurrentWidget(self.pages["dashboard"])

    def _ensure_page(self, page_name):
        if page_name in self.pages:
            return self.pages[page_name]

        if page_name not in self._page_factories:
            return None

        page = self._page_factories[page_name]()
        self.pages[page_name] = page
        self.addWidget(page)
        return page

    def switch_page(self, page_name):
        page = self._ensure_page(page_name)
        if page is not None:
            self.setCurrentWidget(page)