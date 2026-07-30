"""
Shared storage schema constants (CSV).
"""

DEFAULT_DATA_DIR = "aiwaf_data"

WHITELIST_CSV = "whitelist.csv"
BLACKLIST_CSV = "blacklist.csv"
KEYWORDS_CSV = "keywords.csv"
GEO_BLOCKED_COUNTRIES_CSV = "geo_blocked_countries.csv"
PATH_EXEMPTIONS_CSV = "path_exemptions.csv"

CSV_HEADERS = {
    WHITELIST_CSV: ["ip", "added_date"],
    BLACKLIST_CSV: [
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
