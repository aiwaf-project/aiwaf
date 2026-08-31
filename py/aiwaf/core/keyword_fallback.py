"""
Shared fallback keyword storage (file-backed).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict


class KeywordFallbackStore:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self._keywords = defaultdict(int)

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._keywords = defaultdict(int, data)
            except (OSError, json.JSONDecodeError):
                self._keywords = defaultdict(int)

    def _save(self):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(dict(self._keywords), f, indent=2)

    def add(self, keyword: str, count: int = 1):
        self._load()
        self._keywords[keyword] += count
        self._save()

    def top(self, n: int = 10):
        self._load()
        return sorted(self._keywords.items(), key=lambda x: x[1], reverse=True)[:n]
