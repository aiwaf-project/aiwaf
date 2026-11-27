"""
Shared storage schema constants (CSV).
"""

import os

DEFAULT_DATA_DIR = "aiwaf_data"

ALLOWLIST_CSV = "allowlist.csv"
BLOCKLIST_CSV = "blocklist.csv"
KEYWORDS_CSV = "keywords.csv"
GEO_BLOCKED_COUNTRIES_CSV = "geo_blocked_countries.csv"
PATH_EXEMPTIONS_CSV = "path_exemptions.csv"

CSV_HEADERS = {
    ALLOWLIST_CSV: ["ip", "added_date"],
    BLOCKLIST_CSV: [
        "ip",
        "reason",
        "reputation_reason",
        "added_date",
        "extended_request_info",
        "score",
        "offenses",
        "blocked_at",
        "expires_at",
        "duration",
        "permanent",
        "reasons",
    ],
    KEYWORDS_CSV: ["keyword", "added_date"],
    GEO_BLOCKED_COUNTRIES_CSV: ["country", "added_date"],
    PATH_EXEMPTIONS_CSV: ["path", "reason", "added_date"],
}


# Files written under the earlier list names, moved to the current ones on first use.
LEGACY_LIST_FILES = {"whitelist.csv": ALLOWLIST_CSV, "blacklist.csv": BLOCKLIST_CSV}


def move_legacy_list_files(data_dir) -> None:
    """Move whitelist.csv and blacklist.csv in data_dir to allowlist.csv and blocklist.csv.

    A file is moved only when the current one doesn't exist yet, so a directory that has both
    keeps using the current file and leaves the earlier one alone. The engine does the same.
    """
    for legacy, current in LEGACY_LIST_FILES.items():
        target = os.path.join(data_dir, current)
        source = os.path.join(data_dir, legacy)
        if os.path.exists(target) or not os.path.exists(source):
            continue
        try:
            os.rename(source, target)
        except OSError:
            # Another process moved it first; anything else surfaces when the file is used.
            if not os.path.exists(target):
                raise


# Compatibility aliases for callers using the previous terminology.
WHITELIST_CSV = ALLOWLIST_CSV
BLACKLIST_CSV = BLOCKLIST_CSV
