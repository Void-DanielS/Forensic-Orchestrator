"""Web browsers: Chromium (Chrome, Edge, Brave, Opera, Samsung Internet), Firefox, Safari.

Chromium's ``History`` has no extension; the path rule also accepts Android app data dirs.
"""
from src.agents.parsers.base import Col, Lookup, TableSpec

_CHROMIUM = r"(^|/)History$"

SPECS = [
    TableSpec("chrome.url", _CHROMIUM, ("urls",),
              (Col("url", ("url",)), Col("title", ("title",)), Col("visit_count", ("visit_count",)),
               Col("last_visit", ("last_visit_time",), "webkit_us")),
              require=("url",), text=("url",), max_rows=2000,
              claim="Visited {url} ('{title}') {visit_count} time(s), last {last_visit_utc}",
              description="Chromium history: urls"),
    TableSpec("chrome.search", _CHROMIUM, ("keyword_search_terms",),
              (Col("term", ("term",)), Col("url_id", ("url_id",), cite=False)),
              require=("term",), text=("term",),
              lookups=(Lookup("url", ("urls",), ("id",), ("url",), "url_id"),),
              claim="Searched for '{term}' ({url})",
              description="Chromium history: omnibox / search-engine terms"),
    TableSpec("chrome.download", _CHROMIUM, ("downloads",),
              (Col("target_path", ("target_path", "current_path")), Col("start", ("start_time",), "webkit_us"),
               Col("received_bytes", ("received_bytes",)), Col("total_bytes", ("total_bytes",)),
               Col("tab_url", ("tab_url", "referrer")), Col("id", ("id",), cite=False)),
              require=("target_path",), text=("target_path",),
              lookups=(Lookup("url", ("downloads_url_chains",), ("id",), ("url",), "id"),),
              claim="Downloaded {url} to {target_path} at {start_utc} ({received_bytes} bytes)",
              description="Chromium history: downloads"),
    TableSpec("firefox.url", r"(^|/)places\.sqlite$", ("moz_places",),
              (Col("url", ("url",)), Col("title", ("title",)),
               Col("visit_count", ("visit_count", "visit_count_local")),
               Col("last_visit", ("last_visit_date", "last_visit_date_local"), "unix_auto")),
              require=("url", "visit_count"), text=("url",), max_rows=2000, where="{col:visit_count} > 0",
              claim="Visited {url} ('{title}') {visit_count} time(s), last {last_visit_utc}",
              description="Firefox history (places.sqlite, moz_places)"),
    TableSpec("safari.visit", r"(^|/)History\.db$", ("history_visits",),
              (Col("visit_time", ("visit_time",), "cocoa_s"), Col("title", ("title",)),
               Col("history_item", ("history_item",), cite=False)),
              require=("visit_time", "history_item"), max_rows=2000,
              lookups=(Lookup("url", ("history_items",), ("id",), ("url",), "history_item"),),
              text=("url",),
              claim="Safari visit to {url} ('{title}') at {visit_time_utc}",
              description="Safari history (History.db, history_visits + history_items)"),
]
